import math

import cv2
import numpy as np
import torch
import torch.nn as nn

from nanodet.util import bbox2distance, keypoints2distance, distance2bbox, distance2keypoints, multi_apply, overlay_bbox_cv

from ...data.transform.warp import warp_boxes, warp_keypoints_2d
from ..loss.gfocal_loss import DistributionFocalLoss, QualityFocalLoss
from ..loss.iou_loss import GIoULoss
from ..loss.oks_loss import OksLoss                         # TODO: 引入 OksLoss
from ..loss.wing_loss import WingLoss
from ..module.conv import ConvModule, DepthwiseConvModule
from ..module.init_weights import normal_init
from ..module.nms import multiclass_nms
from .assigner.dsl_assigner import DynamicSoftLabelAssigner
from .gfl_head import Integral, reduce_mean

# NanoDetPlusHead 参考了 GFL_Head, 使用 gfl_head.py 中的工具函数
# 这里的修改暂时适用的是 RMer 的四点模型, 有一点很特别
# 1  4
# 2  3
# 其中四个点的顺序固定, 
class NanoDetPlusHead(nn.Module):
    """Detection head used in NanoDet-Plus.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        loss (dict): Loss config.
        input_channel (int): Number of channels of the input feature.
        feat_channels (int): Number of channels of the feature.
            Default: 96.
        stacked_convs (int): Number of conv layers in the stacked convs. 堆叠的卷积层数
            Default: 2.
        kernel_size (int): Size of the convolving kernel. Default: 5.
        strides (list[int]): Strides of input multi-level feature maps. 下采样步长
            Default: [8, 16, 32].
        conv_type (str): Type of the convolution.   是否使用深度可分离卷积
            Default: "DWConv".
        norm_cfg (dict): Dictionary to construct and config norm layer. 
            Default: dict(type='BN').
        reg_max (int): The maximal value of the discrete set. Default: 7.
        activation (str): Type of activation function. Default: "LeakyReLU".
        assigner_cfg (dict): Config dict of the assigner. Default: dict(topk=13).
    """

    def __init__(       # 这里的参数和配置文件里的匹配
        self,
        num_classes,
        num_keypoints,
        loss,
        input_channel,
        feat_channels=96,
        stacked_convs=2,
        kernel_size=5,
        strides=[8, 16, 32],
        conv_type="DWConv",
        norm_cfg=dict(type="BN"),
        reg_max=7,
        activation="LeakyReLU",         # 激活函数
        assigner_cfg=dict(topk=13),     # 分配器参数
        **kwargs
    ):
        super(NanoDetPlusHead, self).__init__()
        self.num_classes = num_classes
        # TODO: 添加关键点个数
        self.num_keypoints = num_keypoints
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.kernel_size = kernel_size
        self.strides = strides
        self.reg_max = reg_max
        self.activation = activation

        # 使用深度可分离卷积
        self.ConvModule = ConvModule if conv_type == "Conv" else DepthwiseConvModule

        self.loss_cfg = loss
        self.norm_cfg = norm_cfg

        # 动态分配器
        self.assigner = DynamicSoftLabelAssigner(**assigner_cfg)
        
        # 根据输出的框分布进行积分, 得到最终的位置值
        self.distribution_project = Integral(self.reg_max, self.num_keypoints)   # 添加 self.num_keypoints 的输出

        # 类别相关
        self.loss_qfl = QualityFocalLoss(
            beta=self.loss_cfg.loss_qfl.beta,
            loss_weight=self.loss_cfg.loss_qfl.loss_weight,
        )
        self.loss_dfl = DistributionFocalLoss(
            loss_weight=self.loss_cfg.loss_dfl.loss_weight
        )
        self.loss_bbox = GIoULoss(loss_weight=self.loss_cfg.loss_bbox.loss_weight)
        
        # TODO: 添加关键点损失函数, 选用 OksLoss 损失函数
        self.loss_keypoints = OksLoss(loss_weight=self.loss_cfg.loss_keypoints.loss_weight)
        
        # 调用 _init_layers 生成对应的卷积结构
        self._init_layers()
        # 调用 init_weights 对网络权重参数进行初始化
        self.init_weights()

    # 初始化卷积层, 
    def _init_layers(self):
        # 包括了多个尺寸的输出头
        # 每部分包括了前置的卷积层 cls_convs 以及 最终的输出部分 gfl_cls
        self.cls_convs = nn.ModuleList()
        for _ in self.strides:
            cls_convs = self._buid_not_shared_head()    # cls 和 reg 共享这些参数
            self.cls_convs.append(cls_convs)

        # 不同步长都有自己的卷积, 为每个头增加 gfl 卷积
        self.gfl_cls = nn.ModuleList(
            [
                nn.Conv2d(
                    self.feat_channels,                                 # 输入通道数
                    # TODO: 添加关键点的输出 2 * num_keypoints 部分
                    # 这里的四点也参考了 bbox 的两点, 参考了积分输出, 也可以使用的别的方式
                    # 这里的四点
                    self.num_classes + 4 * (self.reg_max + 1) + 2 * self.num_keypoints * (self.reg_max + 1),
                    1,  # 卷积核的大小为 1
                    padding=0,
                )
                for _ in self.strides
            ]
        )

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            # 第一层 in_channels, 随后的层为 feat_channels
            chn = self.in_channels if i == 0 else self.feat_channels
            # 只改变通道数, 并未改变 feature map 的尺寸
            cls_convs.append(
                self.ConvModule(
                    chn,                            # 输入通道
                    self.feat_channels,             # 输出通道
                    self.kernel_size,               # 卷积核
                    stride=1,                       # 步长
                    padding=self.kernel_size // 2,  # 卷积填充
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    activation=self.activation,
                )
            )
        return cls_convs

    # 初始化权重
    def init_weights(self):
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        # init cls head with confidence = 0.01
        bias_cls = -4.595
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
        print("Finish initialize NanoDet-Plus Head.")

    # forward, 有助于理解网络的 head 的结构
    def forward(self, feats):
        if torch.onnx.is_in_onnx_export():
            return self._forward_onnx(feats)
        outputs = []

        # 遍历不同的尺寸
        for feat, cls_convs, gfl_cls in zip(
            feats,
            self.cls_convs,
            self.gfl_cls,
        ):
            # 前半部分
            for conv in cls_convs:
                feat = conv(feat)
            # 后半部分得到具体的输出
            output = gfl_cls(feat)
            # [batch, channel, w, h] -> [batch, channel, w*h]
            outputs.append(output.flatten(start_dim=2))
        # 拼接各个尺寸的输出, 为 [batch, w1*h1+w2*h2+w3*h3, channel]
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        return outputs

    # 损失函数, 重要, 训练时使用
    # preds 为网络的输出, gt_meta 为标注信息
    def loss(self, preds, gt_meta, aux_preds=None):
        """Compute losses.
        Args:
            preds (Tensor): Prediction output.
            gt_meta (dict): Ground truth information.
            aux_preds (tuple[Tensor], optional): Auxiliary head prediction output.

        Returns:
            loss (Tensor): Loss tensor.
            loss_states (dict): State dict of each loss.
        """
        device = preds.device
        batch_size = preds.shape[0]
        # gt_xxx: 标注信息
        gt_bboxes = gt_meta["gt_bboxes"]
        gt_labels = gt_meta["gt_labels"]

        # TODO: 添加 keypoints, 包含了可见性, 在后面 loss -> assign -> sample 中去除了 vis
        gt_keypoints = gt_meta["gt_keypoints"]

        gt_bboxes_ignore = gt_meta["gt_bboxes_ignore"]
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(batch_size)]

        input_height, input_width = gt_meta["img"].shape[2:]
        # 获取生成各个尺寸的 feature map size
        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in self.strides
        ]
        # 根据 feature map size 生成锚点
        # get grid cells of one image
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                batch_size,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate(self.strides)
        ]
        center_priors = torch.cat(mlvl_center_priors, dim=1)

        # 对 channels 进行拆分得到 cls, bbox + keypoints 两部分
        # TODO: 添加关键点部分
        cls_preds, reg_preds = preds.split(
            [self.num_classes, (4 + 2 * self.num_keypoints) * (self.reg_max + 1)], dim=-1
        )
        # 对输出进行积分求和得到点
        # distance_preds, 直接对bbox的两个点和keypoints的四点都进行
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        
        # 将 boungding box 部分的输出进行解码, 得到 bbox
        decoded_bboxes = distance2bbox(center_priors[..., :2], dis_preds[..., :4])
        # 添加 keypoints 部分
        decoded_keypoints = distance2keypoints(center_priors[..., :2], dis_preds[..., 4:])    

        # 进行标签分配
        if aux_preds is not None:
            # use auxiliary head to assign
            aux_cls_preds, aux_reg_preds = aux_preds.split(
                [self.num_classes, (4 + 2 * self.num_keypoints) * (self.reg_max + 1)], dim=-1
            )
            aux_dis_preds = (
                self.distribution_project(aux_reg_preds) * center_priors[..., 2, None]
            )
            aux_decoded_bboxes = distance2bbox(center_priors[..., :2], aux_dis_preds[..., :4])
            # 添加修改部分
            aux_decoded_keypoints = distance2keypoints(center_priors[..., :2], aux_dis_preds[..., 4:])
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                aux_cls_preds.detach(),
                center_priors,
                aux_decoded_bboxes.detach(),
                aux_decoded_keypoints.detach(),
                gt_bboxes,
                gt_labels,
                gt_keypoints,
                gt_bboxes_ignore,
            )
        else:
            # use self prediction to assign
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                cls_preds.detach(),
                center_priors,
                decoded_bboxes.detach(),
                decoded_keypoints.detach(),     # TODO: 添加 keypoints 部分
                gt_bboxes,
                gt_labels,
                gt_keypoints,                   # TODO: 
                gt_bboxes_ignore,
            )

        # TODO: 使用 assign 的结果计算损失函数
        loss, loss_states = self._get_loss_from_assign(
            cls_preds, reg_preds, decoded_bboxes, decoded_keypoints, batch_assign_res
        )

        if aux_preds is not None:
            aux_loss, aux_loss_states = self._get_loss_from_assign(
                aux_cls_preds, aux_reg_preds, aux_decoded_bboxes, aux_decoded_keypoints, batch_assign_res
            )
            loss = loss + aux_loss
            for k, v in aux_loss_states.items():
                loss_states["aux_" + k] = v
        return loss, loss_states

    # 根据分配结果计算损失函数
    def _get_loss_from_assign(self, cls_preds, reg_preds, decoded_bboxes, decoded_keypoints, assign):
        device = cls_preds.device
        (
            labels,                     # 标签 (num_priros)
            label_scores,               # (num_priros)
            label_weights,              # (num_priros)
            bbox_targets,               # (num_priros, 4)
            keypoints_targets,          # (num_priors, 2*num_keypoints)
            dist_bbox_targets,          # (num_priros, 4)
            dist_keypoints_targets,     # (num_priors, 2*num_keypoints)
            num_pos,                    # 正样本个数
        ) = assign
        # 样本总数
        num_total_samples = max(
            reduce_mean(torch.tensor(sum(num_pos)).to(device)).item(), 1.0
        )

        # [b, num_priors]
        labels = torch.cat(labels, dim=0)
        label_scores = torch.cat(label_scores, dim=0)
        label_weights = torch.cat(label_weights, dim=0)
        bbox_targets = torch.cat(bbox_targets, dim=0)
        keypoints_targets = torch.cat(keypoints_targets, dim=0)
        # 将 bbox 和 keypoints 的 dist 进行合并
        dist_bbox_targets = torch.cat(dist_bbox_targets, dim=0)       
        dist_keypoints_targets = torch.cat(dist_keypoints_targets, dim=0) 
        dist_targets = torch.cat([dist_bbox_targets, dist_keypoints_targets], dim=-1)

        # (num_priors, num_classes)
        cls_preds = cls_preds.reshape(-1, self.num_classes)
        # TODO: 将 reg 输出 reshape, 不用拆分, 主要用来计算 dfl
        reg_preds = reg_preds.reshape(-1, (4 + 2 * self.num_keypoints) * (self.reg_max + 1))
        
        # (num_priors, 4)
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)
        # TODO: 将输出的 keypoints 转换成 (-1, 2*num_keypoints) 的 shape
        decoded_keypoints = decoded_keypoints.reshape(-1, 2 * self.num_keypoints)
    
        # Loss  QFL
        loss_qfl = self.loss_qfl(
            cls_preds,
            (labels, label_scores),
            weight=label_weights,
            avg_factor=num_total_samples,
        )

        pos_inds = torch.nonzero(
            (labels >= 0) & (labels < self.num_classes), as_tuple=False
        ).squeeze(1)

        # 正样本的个数 > 0
        if len(pos_inds) > 0:
            weight_targets = cls_preds[pos_inds].detach().sigmoid().max(dim=1)[0]
            bbox_avg_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)

            # 计算 bbox 的损失函数
            loss_bbox = self.loss_bbox(
                decoded_bboxes[pos_inds],   # 输出的正样本    [num_pos, 4]
                bbox_targets[pos_inds],     # 对应的 gt_bbox  [num_pos, 4]
                weight=weight_targets,
                avg_factor=bbox_avg_factor,
            )

            # 计算 keypoints 的损失函数, 调用 Oks
            loss_keypoints = self.loss_keypoints(
                decoded_keypoints[pos_inds].view(-1, self.num_keypoints, 2),
                keypoints_targets[pos_inds].view(-1, self.num_keypoints, 2),
                torch.ones([pos_inds.shape[0], self.num_keypoints]).to(device),
                bbox_targets[pos_inds],
            )
            
            # 计算 dfl 损失函数
            # dist_targets = torch.cat(dist_targets, dim=0)
            loss_dfl = self.loss_dfl(
                reg_preds[pos_inds].reshape(-1, self.reg_max + 1),  # 输出
                dist_targets[pos_inds].reshape(-1),                 # TODO: 简单修改 + 2*num_keypoints
                weight=weight_targets[:, None].expand(-1, 4 + 2 * self.num_keypoints).reshape(-1),
                avg_factor=4.0 * bbox_avg_factor,
            )
        else:
            loss_bbox = reg_preds.sum() * 0             # 0
            loss_keypoints = reg_preds.sum() * 0        # 0
            loss_dfl = reg_preds.sum() * 0              # 0

        loss = loss_qfl + loss_bbox + loss_dfl + loss_keypoints
        loss_states = dict(loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_keypoints=loss_keypoints, loss_dfl=loss_dfl)
        return loss, loss_states

    # 单张图片进行标签分配
    @torch.no_grad()
    def target_assign_single_img(
        self,
        cls_preds,
        center_priors,
        decoded_bboxes,
        decoded_keypoints,      # TODO: 添加 decoded_keypoints
        gt_bboxes,
        gt_labels,
        gt_keypoints,           # TODO: 添加 gt_keypoints
        gt_bboxes_ignore=None,
    ):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            center_priors (Tensor): All priors of one image, a 2D-Tensor with
                shape [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
        """

        device = center_priors.device
        gt_bboxes = torch.from_numpy(gt_bboxes).to(device)
        gt_labels = torch.from_numpy(gt_labels).to(device)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        # TODO: 仿照 gt_bboxes 进行处理
        gt_keypoints = torch.from_numpy(gt_keypoints).to(device)
        gt_keypoints = gt_keypoints.to(decoded_keypoints.dtype)

        if gt_bboxes_ignore is not None:
            gt_bboxes_ignore = torch.from_numpy(gt_bboxes_ignore).to(device)
            gt_bboxes_ignore = gt_bboxes_ignore.to(decoded_bboxes.dtype)

        # 进行标签分配, assign
        # 分配的结果  assigned_gt_inds 的 shape 为 (num_priors)
        # AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        # 借助 bbox 标签分配的结果
        assign_result = self.assigner.assign(
            cls_preds,
            center_priors,
            decoded_bboxes,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
        )

        # TODO: 正负样本
        pos_inds, neg_inds, pos_gt_bboxes, pos_gt_keypoints, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes, gt_keypoints
        )

        num_priors = center_priors.size(0)                  # (num_priors)
        # 坐标的结果
        bbox_targets = torch.zeros_like(center_priors)      # (num_priors, 4)
        keypoints_targets = torch.zeros(num_priors, 2*self.num_keypoints).to(device) # (num_priors, 8)
        
        # dist_targets = torch.zeros_like(center_priors)
        # distance 的结果, 为了方便后续计算损失函数
        dist_bbox_targets = torch.zeros(num_priors, 4).to(device)
        dist_keypoints_targets = torch.zeros(num_priors, 2 * self.num_keypoints).to(device)
        
        # 标签
        labels = center_priors.new_full(
            (num_priors,), self.num_classes, dtype=torch.long
        )
        label_weights = center_priors.new_zeros(num_priors, dtype=torch.float)
        label_scores = center_priors.new_zeros(labels.shape, dtype=torch.float)

        # 一张图片中正样本的个数
        num_pos_per_img = pos_inds.size(0)
        # 正样本和对应的 gt_bbox 的IoU
        pos_ious = assign_result.max_overlaps[pos_inds]

        # 正样本的个数大于 0
        if len(pos_inds) > 0:
            # bbox_targets 为正样本对应 gt_bbox, 存放了对应的 gt_bbox 坐标信息
            bbox_targets[pos_inds, :] = pos_gt_bboxes
            # dist_bbox_targets 为正样本对应的 gt_bbox, 存放了 gt_bbox 转换成距离后的信息)
            dist_bbox_targets[pos_inds, :] = (
                bbox2distance(center_priors[pos_inds, :2], pos_gt_bboxes)
                / center_priors[pos_inds, None, 2]
            )
            dist_bbox_targets = dist_bbox_targets.clamp(min=0, max=self.reg_max - 0.1)

            # keypoints_targets 为正样本对应的 gt_keypoints, 存放的是坐标信息
            keypoints_targets[pos_inds, :] = pos_gt_keypoints
            # dist_keypoints_targets 为正样本对应的 gt_keypoints, 存放的是距离信息 
            dist_keypoints_targets[pos_inds, :] = (
                keypoints2distance(center_priors[pos_inds, :2], pos_gt_keypoints)
                / center_priors[pos_inds, None, 2]
            )
            dist_keypoints_targets = dist_keypoints_targets.clamp(min=0, max=self.reg_max - 0.1)
            
            # 锚点对应的标签     (num_priors)
            labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            # label_score 样本与 gt_bbox 的 iou
            label_scores[pos_inds] = pos_ious
            label_weights[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        
        # 将分配的结果返回, 可以直接用于计算损失函数
        return (
            labels,
            label_scores,
            label_weights,
            bbox_targets,
            keypoints_targets,
            dist_bbox_targets,
            dist_keypoints_targets,
            num_pos_per_img,
        )

    # sample
    def sample(self, assign_result, gt_bboxes, gt_keypoints):
        """Sample positive and negative bboxes."""
        # 每个锚点对应一个样本
        # 正样本的索引  (num_pos)
        pos_inds = (
            torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)    # 作为张量返回
            .squeeze(-1)
            .unique()
        )
        # 负样本的索引  (num_neg)
        neg_inds = (
            torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        # 正样本对应的 gt 索引  (num_pos)
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        # 如果 gt 为空
        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
            # TODO: 补充 pos_gt_keypoints
            pos_gt_keypoints = torch.empty_like(gt_keypoints).view(-1, 2 * self.num_keypoints)
        else:
            # gt 不为空的情况
            if len(gt_bboxes.shape) < 2:            # 如果小于二维, 则调整为二维的
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
            # TODO: 添加 gt_keypoints, 去除其中的可见性信息
            gt_keypoints = gt_keypoints.view(-1, self.num_keypoints, 3)[:,:,:2].reshape(-1, 2 * self.num_keypoints)
            pos_gt_keypoints = gt_keypoints[pos_assigned_gt_inds, :]
        # 返回 正样本的索引(num_pos)、负样本的索引(num_neg), 这两个索引是在 num_priors 中的索引
        # 正样本对应的 gt_bboxes(num_pos, 4)
        # 负样本对应的 gt_keypoints (num_pos, 2*num_keypoints)
        # pos_assigned_gt_inds 为 正样本对应的 gt 索引 (num_pos) , 这个索引是在 num_gts 中的索引
        return pos_inds, neg_inds, pos_gt_bboxes, pos_gt_keypoints, pos_assigned_gt_inds

    # 后处理函数, 真正推理时的处理, 其他格式的处理也要参考这个函数
    # 验证以及推理时需要使用
    def post_process(self, preds, meta):
        """Prediction results post processing. Decode bboxes and rescale
        to original image size.
        Args:
            preds (Tensor): Prediction output.
            meta (dict): Meta info.
        """
        cls_scores, reg_preds = preds.split(
            [self.num_classes, (4 + 2 * self.num_keypoints) * (self.reg_max + 1)], dim=-1
        )
        # TODO: 修改内容
        result_list = self.get_results(cls_scores, reg_preds, meta)
        det_results = {}
        warp_matrixes = (
            meta["warp_matrix"]
            if isinstance(meta["warp_matrix"], list)
            else meta["warp_matrix"]
        )
        img_heights = (
            meta["img_info"]["height"].cpu().numpy()
            if isinstance(meta["img_info"]["height"], torch.Tensor)
            else meta["img_info"]["height"]
        )
        img_widths = (
            meta["img_info"]["width"].cpu().numpy()
            if isinstance(meta["img_info"]["width"], torch.Tensor)
            else meta["img_info"]["width"]
        )
        img_ids = (
            meta["img_info"]["id"].cpu().numpy()
            if isinstance(meta["img_info"]["id"], torch.Tensor)
            else meta["img_info"]["id"]
        )

        for result, img_width, img_height, img_id, warp_matrix in zip(
            result_list, img_widths, img_heights, img_ids, warp_matrixes
        ):
            det_result = {}
            # bbox 4 + keypoints 8 + score 1
            det_outputs, det_labels = result
            det_outputs = det_outputs.detach().cpu().numpy()
            det_outputs[:, :4] = warp_boxes(
                det_outputs[:, :4], np.linalg.inv(warp_matrix), img_width, img_height
            )
            det_outputs[:, 4:-1] = warp_keypoints_2d(
                det_outputs[:, 4:-1], np.linalg.inv(warp_matrix), img_width, img_height
            )
            classes = det_labels.detach().cpu().numpy()
            for i in range(self.num_classes):
                inds = classes == i
                det_result[i] = np.concatenate(
                    [
                        det_outputs[inds, :4].astype(np.float32),   # bbox
                        det_outputs[inds, 4:4 + 2*self.num_keypoints].astype(np.float32),   # keypoints
                        det_outputs[inds, 4 + 2*self.num_keypoints:4 + 2*self.num_keypoints + 1].astype(np.float32),    # score
                    ],
                    axis=1,
                ).tolist()
            det_results[img_id] = det_result    # 4 + 2*num_keypoints + 1
        return det_results

    def show_result(
        self, img, dets, class_names, score_thres=0.3, show=True, save_path=None
    ):
        result = overlay_bbox_cv(img, dets, class_names, score_thresh=score_thres)
        # 在服务器上关闭, 没有窗口, 选择设置 --save_result 来进行保存
        # if show:
        #     cv2.imshow("det", result)
        return result

    # 修改：results
    def get_results(self, cls_preds, reg_preds, img_metas):
        """Decode the outputs to bboxes.
        Args:
            cls_preds (Tensor): Shape (num_imgs, num_points, num_classes).
            reg_preds (Tensor): Shape (num_imgs, num_points, 4 * (regmax + 1)).
            img_metas (dict): Dict of image info.

        Returns:
            results_list (list[tuple]): List of detection bboxes and labels.
        """
        device = cls_preds.device
        b = cls_preds.shape[0]
        input_height, input_width = img_metas["img"].shape[2:]
        input_shape = (input_height, input_width)

        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in self.strides
        ]
        # get grid cells of one image
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                b,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate(self.strides)
        ]
        center_priors = torch.cat(mlvl_center_priors, dim=1)
        # 对输出进行积分解码
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        # 根据 dis_preds 得到 bbox
        bboxes = distance2bbox(center_priors[..., :2], dis_preds[..., :4], max_shape=input_shape)
        keypoints = distance2keypoints(center_priors[..., :2], dis_preds[..., 4:], max_shape=input_shape)
        scores = cls_preds.sigmoid()
        result_list = []
        for i in range(b):
            # add a dummy background class at the end of all labels
            # same with mmdetection2.0
            score, bbox, points = scores[i], bboxes[i], keypoints[i]
            padding = score.new_zeros(score.shape[0], 1)
            score = torch.cat([score, padding], dim=1)
            results = multiclass_nms(
                bbox,
                points,                 # TODO: 添加输入 points
                score,
                score_thr=0.05,
                nms_cfg=dict(type="nms", iou_threshold=0.6, class_agnostic=True),   # nms 处理
                max_num=100,
                num_keypoints=self.num_keypoints
            )
            result_list.append(results)
        return result_list

    # 获取 feature map 的锚点, 步长和 w h 搭配
    def get_single_level_center_priors(
        self, batch_size, featmap_size, stride, dtype, device
    ):
        """Generate centers of a single stage feature map.
        Args:
            batch_size (int): Number of images in one batch.
            featmap_size (tuple[int]): height and width of the feature map
            stride (int): down sample stride of the feature map
            dtype (obj:`torch.dtype`): data type of the tensors
            device (obj:`torch.device`): device of the tensors
        Return:
            priors (Tensor): center priors of a single level feature map.
        """
        h, w = featmap_size
        x_range = (torch.arange(w, dtype=dtype, device=device)) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device)) * stride
        y, x = torch.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0],), stride)
        proiors = torch.stack([x, y, strides, strides], dim=-1)
        return proiors.unsqueeze(0).repeat(batch_size, 1, 1)

    # onnx 前向推理
    def _forward_onnx(self, feats):
        """only used for onnx export"""
        outputs = []
        for feat, cls_convs, gfl_cls in zip(
            feats,
            self.cls_convs,
            self.gfl_cls,
        ):
            for conv in cls_convs:
                feat = conv(feat)
            output = gfl_cls(feat)
            # TODO: 修改 forward_onnx
            cls_pred, reg_pred = output.split(
                [self.num_classes, (4 + 2 * self.num_keypoints) * (self.reg_max + 1)], dim=1
            )
            cls_pred = cls_pred.sigmoid()
            out = torch.cat([cls_pred, reg_pred], dim=1)
            outputs.append(out.flatten(start_dim=2))
        return torch.cat(outputs, dim=2).permute(0, 2, 1)

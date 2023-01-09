import math

import cv2
import numpy as np
import torch
import torch.nn as nn

from nanodet.util import bbox2distance, distance2bbox, multi_apply, overlay_bbox_cv

from ...data.transform.warp import warp_boxes
from ..loss.gfocal_loss import DistributionFocalLoss, QualityFocalLoss
from ..loss.iou_loss import GIoULoss
from ..loss.wing_loss import WingLoss                   # TODO : 新增wingloss损失函数
from ..module.conv import ConvModule, DepthwiseConvModule
from ..module.init_weights import normal_init
from ..module.nms import multiclass_nms
from .assigner.dsl_assigner import DynamicSoftLabelAssigner
from .gfl_head import Integral, reduce_mean

# ghost-pan部分，给到head的部分的输入是， [1x96x40x40], [1x96x20x20], [1x96x10x10], [1x96x5x5]
class NanoDetPlusHead(nn.Module):
    """Detection head used in NanoDet-Plus.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        loss (dict): Loss config.
        input_channel (int): Number of channels of the input feature.
        feat_channels (int): Number of channels of the feature.
            Default: 96.
        stacked_convs (int): Number of conv layers in the stacked convs.
            Default: 2.
        kernel_size (int): Size of the convolving kernel. Default: 5.
        strides (list[int]): Strides of input multi-level feature maps.
            Default: [8, 16, 32].
        conv_type (str): Type of the convolution.
            Default: "DWConv".
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN').
        reg_max (int): The maximal value of the discrete set. Default: 7.       # 可以看gfl论文
        activation (str): Type of activation function. Default: "LeakyReLU".
        assigner_cfg (dict): Config dict of the assigner. Default: dict(topk=13).
    """

    # 当前的输出是 [1,2125,112] 其中2125为预期的个数，112为 [80+(7+1)x4]   要新增 8 ，变成 120
    def __init__(
        self,
        num_classes,
        loss,
        input_channel,
        feat_channels=96,
        stacked_convs=2,
        kernel_size=5,
        strides=[8, 16, 32],                 # config中的参数是 [8, 16, 32, 64]
        conv_type="DWConv",
        norm_cfg=dict(type="BN"),
        reg_max=7,
        activation="LeakyReLU",
        assigner_cfg=dict(topk=13),
        **kwargs
    ):
        super(NanoDetPlusHead, self).__init__()
        self.num_classes = num_classes                      # 36个类别
        self.in_channels = input_channel                    # 96
        self.feat_channels = feat_channels                  # 96
        self.stacked_convs = stacked_convs                  # 2
        self.kernel_size = kernel_size
        self.strides = strides                              # [98, 16, 32, 64]
        self.reg_max = reg_max                              # 7
        self.activation = activation
        self.ConvModule = ConvModule if conv_type == "Conv" else DepthwiseConvModule

        self.loss_cfg = loss
        self.norm_cfg = norm_cfg

        self.assigner = DynamicSoftLabelAssigner(**assigner_cfg)    # 动态标间分配器
        self.distribution_project = Integral(self.reg_max)

        # 联合分类和框的质量估计表示
        self.loss_qfl = QualityFocalLoss(
            beta=self.loss_cfg.loss_qfl.beta,
            loss_weight=self.loss_cfg.loss_qfl.loss_weight,
        )

        # 初始化参数中reg_max的由来
        self.loss_dfl = DistributionFocalLoss(
            loss_weight=self.loss_cfg.loss_dfl.loss_weight
        )

        # iou的一种改进GIOU
        self.loss_bbox = GIoULoss(loss_weight=self.loss_cfg.loss_bbox.loss_weight)
        # wing_loss , 关键点检测损失函数
        self.wing_loss = WingLoss()                                    # TODO: head增加WingLoss损失函数

        self._init_layers()
        self.init_weights()

    # 生成网络需要的结构：一个head对应，四个卷积cls_convs + 一个gfl_cls部分，nanodet-plus有4个head
    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        for _ in self.strides:
            # 为每个stride创建一个head，cls和reg共享这些参数
            cls_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)    # 四层

        # 为每个头增加gfl卷积 输出（1x1的卷积，改变通道数）
        self.gfl_cls = nn.ModuleList(                           # nanodet的head采用的是统一的输出，直接一个112既包括了类别也包括了坐标相关内容
            [
                nn.Conv2d(
                    self.feat_channels,                         # 特征通道数
                    self.num_classes + 4 * (self.reg_max + 1) + 8,  # num_classes个通道用于一寸类别分数，还有4*(reg_max+1)来回归位置，     TODO 新增8个通道，作为4个点的输出。为了稳妥，就不删除原有的bbox检测了
                    1,
                    padding=0,
                )
                for _ in self.strides                           # 每个尺度增加一个gfl卷积
            ]
        )

    # 利用上面的初始化的内容，来构建网络，生成网络检测head，一个stride对应的head的前半部分（4个卷积）
    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            # 第一层要和PAN的输出对齐通道
            chn = self.in_channels if i == 0 else self.feat_channels
            # 后面就是卷积了
            cls_convs.append(
                self.ConvModule(
                    chn,
                    self.feat_channels,
                    self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,  # 使得输入输出有相同尺寸
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    activation=self.activation,
                )
            )
        return cls_convs

    # 采用norm初始化网络参数
    def init_weights(self):
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        # init cls head with confidence = 0.01
        bias_cls = -4.595
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
        print("Finish initialize NanoDet-Plus Head.")

    # 前向推理部分
    def forward(self, feats):   
        if torch.onnx.is_in_onnx_export():
            return self._forward_onnx(feats)
        # 输出默认有4份，是一个list
        outputs = []
        # feats来自FPN，且组数和self.cls_convs, self.gfl_cls 的数量一致
        for feat, cls_convs, gfl_cls in zip(
            feats,
            self.cls_convs,
            self.gfl_cls,
        ):
            # 对每组feat进行前向推理操作
            # 四个卷积的部分
            for conv in cls_convs:
                feat = conv(feat)
            # 一个gfl_cls部分
            output = gfl_cls(feat)
            # 所有head的输出会在展平后拼成一个tensor，方便后处理
            # output是一个四维tensor，第一维长度为1
            # 长度W，宽度H，即feat的大小，高为 80+4*(reg_max+1) 即 cls和reg
            # 按照第三个维度展平，就是排成一个长度为 W*H 的tensor，另一个维度是输出的cls和reg
            outputs.append(output.flatten(start_dim=2))                 # 1x112x(W*H)  ->  1x120x(W*H)
        # 把不同head的输出交换一下维度排列顺序，全部拼在一起
        # 按照第三维拼接，就是     1x112x2125 -> 2x120x2125
        # [b, c, w*h]->[b, w*h, c]      [1, 2125, 112]                  [batchsize, w*h, c]     其中channel为输出的部分
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        return outputs                                                  # 得到最终的输出部分

    # gt_meta 就是用户标注的数据，可以认为就是标签文件的内容， preds就是    [1, 2125, 112]         [batchsize, w*h, c]
    def loss(self, preds, gt_meta, aux_preds=None):
        """Compute losses.
        Args:
            preds (Tensor): Prediction output.                  head部分的输出
            gt_meta (dict): Ground truth information.           包含gt位置和标签的字典, 还有原图像的数据
            aux_preds (tuple[Tensor], optional): Auxiliary head prediction output.  如果AGM还没有detach, 会用AGM的输出进行标签分配

        Returns:
            loss (Tensor): Loss tensor.
            loss_states (dict): State dict of each loss.
        """
        # 设备
        device = preds.device
        # 得到本次loss计算的batch数，pred是3维的tensor
        batch_size = preds.shape[0] 
        # 把gt相关的数据分离出来，这两个数据都是list，长度为batchsize的大小
        # 每个list都包含他们各自对应的图像上的gt和label
        gt_bboxes = gt_meta["gt_bboxes"]    # 检验框             最终应该是 [batchsize, num_gts, 4] 大小
        gt_labels = gt_meta["gt_labels"]    # 类别              [batchsize, num_gts]
        gt_points = gt_meta["gt_points"]                        # TODO： 新增部分gt_labels = gt_meta["gt_labels"]   数据集中需要新增的四个点 [batchsize, num_gts, 8]

        gt_bboxes_ignore = gt_meta["gt_bboxes_ignore"]
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(batch_size)]

        # img信息提取长宽，（就是标签中的一张图片的大小）
        # 所有图片都会在前处理中被resize成网络的输入大小，不足则直接加zero padding
        input_height, input_width = gt_meta["img"].shape[2:]
        
        # 如果修改了输入或者采样率，输入无法被stride整除，所以要用ceil取整
        # 因为稍后要布置priors，这里要计算出feature map的大小   [40, 40] [20, 20] [10, 10] [5, 5]
        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in self.strides
        ]

        # get grid cells of one image       为了方便计算bbox的损失函数准备
        # 在不同大小的stride上放置一组prior，默认四个检测头也就是四个不同尺寸的stride
        # 最后返回的tensor维度是[batchsize, strideW*strideH, 4] 
        # 其中每一个都是[x, y, strideH, strideW]的结构，当featuremap不是正方形的时候两个stride不相等
        # 相当于生成了一堆锚点（图像中横纵一定步长，生成一些列点，anchor-free 会生成一系列锚点）
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

        # 按照第二个维度拼接后的prior的维度是[batchsize, 40x40+20x20+10x10+5x5, 4]
        # 其中四个值为[cx, cy, strideW, strideH]，横纵像素坐标，以及步长
        center_priors = torch.cat(mlvl_center_priors, dim=1)
        
        # 预测部分：    把预测值拆分成分类和框回归
        # cls_preds的维度是[batchsize, 2125*class_num], reg_pred是[batchsize, 2125*4*(reg_max+1)]
        cls_preds, reg_preds, pts_preds = preds.split(               # TODO 新增pts_preds部分，split出四个角点回归的部分
            [self.num_classes, 4 * (self.reg_max + 1), 8], dim=-1
        )
        # cls_preds : [1, 2125, 80]   
        # reg_preds : [1, 2125, 32]
        # 相应的 要求，pts_preds : [1, 2125, 8]

        # 对reg_preds进行积分得到位置预测，reg_preds 表示的是一条边的离散分布, 每个锚点预测四条边的距离
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]

        # 把[dl,dr,dt,db] 根据prior的为位置转换成框的左上角和右下角点，方便计算iou，   anchor-free（没有锚框，但是有锚点） 每个锚点对应一个
        decoded_bboxes = distance2bbox(center_priors[..., :2], dis_preds)

        # 如果启用了辅助训练模块，将用辅助训练的结果进行`标签分配`，
        if aux_preds is not None:  

            # use auxiliary head to assign
            aux_cls_preds, aux_reg_preds = aux_preds.split(
                [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
            )

            # 对reg_preds积分得到预测位置，reg_preds 表示的是一条边的离散分布
            aux_dis_preds = (
                self.distribution_project(aux_reg_preds) * center_priors[..., 2, None]
            )

            # 把距离转换成左上角和右下角
            aux_decoded_bboxes = distance2bbox(center_priors[..., :2], aux_dis_preds)

            # 每次给一张图片进行分配，应该是为了避免显存溢出
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                aux_cls_preds.detach(),         # 类别预测
                center_priors,                  # [cx,cy,strideW,strideH] 中心点
                aux_decoded_bboxes.detach(),    # 预测框 的 左上角、右下角坐标 （一共strideW*strideH个框）
                gt_bboxes,                      # 真实框
                gt_labels,                      # 真实类别
                gt_points,                      # TODO： 对于AGM也增加gt_points
                gt_bboxes_ignore,
            )
        else:
            # 如果 没有启用AGM辅助训练
            # multi_apply将参数中的函数作用在后面的每一个可迭代对象上，一次处理批量数据
            # use self prediction to assign
            # target_assign_single_img 一次只能分配一张图片
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                cls_preds.detach(),
                center_priors,
                decoded_bboxes.detach(),
                gt_bboxes,
                gt_labels,
                gt_points,                  # TODO: 修改target_assign_single_img函数，新增gt_points部分
                gt_bboxes_ignore,
            )
        
        # 根据·分配结果·计算loss 
        loss, loss_states = self._get_loss_from_assign(
            cls_preds, reg_preds, pts_preds, decoded_bboxes, batch_assign_res     # TODO: 新增pts_preds输入
        )

        # 加入 `辅助训练模块的loss` ，这可以让网络在初期收敛的更快
        if aux_preds is not None:
            aux_loss, aux_loss_states = self._get_loss_from_assign(
                aux_cls_preds, aux_reg_preds, pts_preds, aux_decoded_bboxes, batch_assign_res
            )
            loss = loss + aux_loss
            for k, v in aux_loss_states.items():
                loss_states["aux_" + k] = v

        return loss, loss_states

    # prior就是框分布的回归起点，将以prior的位置作为目标中心，预测四个值形成检验框, assign target_assign_single_img的结果
    def _get_loss_from_assign(self, cls_preds, reg_preds, pts_preds, decoded_bboxes, assign):       # TODO: 参数中新增了 pts_preds ,pts_targets
                                                                                                                # 这里增加pts_targets, 主要是因为在assign中没有
        device = cls_preds.device
        (
            labels,
            label_scores,
            label_weights,
            bbox_targets,
            pts_targets,                                             # TODO: 新增部分
            dist_targets,
            num_pos,
        ) = assign

        # 因为要对整个batch进行平均，因此，在这里计算出这次分配的总正样本数，用于稍后的weight_loss
        num_total_samples = max(
            reduce_mean(torch.tensor(sum(num_pos)).to(device)).item(), 1.0
        )

        # 为了一次性处理一个batch的数据，把每个结果都拼接起来
        # labels 和 label_score 都是[batchsize*2125], bbox_targets 是 [batchsize*2125, 4*(reg_max+1)]
        labels = torch.cat(labels, dim=0)
        label_scores = torch.cat(label_scores, dim=0)
        label_weights = torch.cat(label_weights, dim=0)
        bbox_targets = torch.cat(bbox_targets, dim=0)
        pts_targets = torch.cat(pts_targets, dim=0)                     # TODO: 新增部分 pts_targets 
        
        # 把预测结果和检验框都reshape成和batch对应的形状
        cls_preds = cls_preds.reshape(-1, self.num_classes)
        reg_preds = reg_preds.reshape(-1, 4 * (self.reg_max + 1))
        pts_preds = pts_preds.reshape(-1, 8)                            # TODO: 新增部分 pts_preds
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)
        
        # 计算quality focal loss
        # 利用iou联合了质量估计和分类表示，和软标签计算相似
        loss_qfl = self.loss_qfl(
            cls_preds,                                                  # cls_preds部分的损失函数
            (labels, label_scores),
            weight=label_weights,
            avg_factor=num_total_samples,
        )
        
        # 获取对应的label标签（把当前batch的所有index交给pos_inds）
        # tensor 中常用逻辑判断语句生成mask掩膜，元素中符合者编程True，反之为Flase
        pos_inds = torch.nonzero(
            (labels >= 0) & (labels < self.num_classes), as_tuple=False
        ).squeeze(1)

        # 结果label为空，说明没有分配任何标签，则不需要计算iou loss
        if len(pos_inds) > 0:
            # 计算用于weight_reduce的参数，weight_target的长度和被分配了gt的prior的数量相同
            # sigmoid后取最大值得到的就是该prior输出的类别分数
            weight_targets = cls_preds[pos_inds].detach().sigmoid().max(dim=1)[0]
            # 同步GPU上的其他worker，获得此参数
            bbox_avg_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)

            # 计算GIoU损失，加入了weighted_loss
            loss_bbox = self.loss_bbox(
                decoded_bboxes[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_targets,
                avg_factor=bbox_avg_factor,
            )

            loss_pts = self.wing_loss(
                pts_preds[pos_inds].reshape(-1, 8),
                pts_targets[pos_inds]
            )                                                               # TODO: 新增点的损失函数
            
            # 同样拼接起来方便批量计算
            dist_targets = torch.cat(dist_targets, dim=0)
            # 计算Distribution focal loss
            loss_dfl = self.loss_dfl(
                reg_preds[pos_inds].reshape(-1, self.reg_max + 1),
                dist_targets[pos_inds].reshape(-1),
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0 * bbox_avg_factor,
            )

        else:
            loss_bbox = reg_preds.sum() * 0
            loss_dfl = reg_preds.sum() * 0
            loss_pts = pts_preds.sum() * 0                                  # TODO: 新增的部分
            
        
        # 计算损失函数
        loss = loss_qfl + loss_bbox + loss_dfl + loss_pts           # TODO: 在计算的部分新增loss_pts部分
        loss_states = dict(loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl, loss_pts=loss_pts)
        return loss, loss_states

    # 标签分配时的运算不会被记录，只是在计算cost并进行匹配，需要特别注意，这个函数只为一张图片，即一个样本进行标签分配
    @torch.no_grad()
    def target_assign_single_img(
        self,
        cls_preds,                          # [2125, 80]
        center_priors,                      
        decoded_bboxes,
        gt_bboxes,                          # [num_gts, 4]                     
        gt_labels,                          # [num_gts]
        gt_points,                                                                       # TODO： 新增gt_points部分
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
        gt_bboxes = torch.from_numpy(gt_bboxes).to(device)  # [num_gts, 4]
        gt_points = torch.from_numpy(gt_points).to(device)                      # TODO :新增部分“仿照gt_points"     [num_gts, 8]
        gt_labels = torch.from_numpy(gt_labels).to(device)  # [num_gts]
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)

        if gt_bboxes_ignore is not None:
            gt_bboxes_ignore = torch.from_numpy(gt_bboxes_ignore).to(device)
            gt_bboxes_ignore = gt_bboxes_ignore.to(decoded_bboxes.dtype)

        # 我在修改的时候：对于assign没有修改，标签分配仍旧按照原来的iou等计算，利用标签分配的结果直接对points提供

        # class的输出要映射到0-1之间, head构建conv layer 可以发现最后的分类没有激活函数
        assign_result = self.assigner.assign(
            cls_preds.sigmoid(),
            center_priors,
            decoded_bboxes,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
        )

        # 调用采样函数，获得正负样本
        pos_inds, neg_inds, pos_gt_bboxes, pos_gt_points, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes, gt_points                         # TODO : 新增gt_points 参数，以及pos_gt_points输出
        )

        num_priors = center_priors.size(0)                          # prior的个数
        bbox_targets = torch.zeros_like(center_priors)
        pts_targets = torch.zeros(num_priors, 8).to(device)    # TODO：新增，points_targets，但是要注意大小和尺寸， center_points的大小为[num_priors, 4]
        dist_targets = torch.zeros_like(center_priors)

        # 把label扩充成 one - hot 向量
        labels = center_priors.new_full(
            (num_priors,), self.num_classes, dtype=torch.long
        )

        # No target
        label_weights = center_priors.new_zeros(num_priors, dtype=torch.float)
        label_scores = center_priors.new_zeros(labels.shape, dtype=torch.float)
        
        # 当前分配到这个图片上的正样本数
        num_pos_per_img = pos_inds.size(0)
        
        # 把分配到了gt的那些prior预测的检验框和gt的iou算出来，用于QFL计算
        pos_ious = assign_result.max_overlaps[pos_inds]

        if len(pos_inds) > 0:
            
            # bbox_targets就是最终用来和gt计算回归损失的东西，维度为[2125, 4]
            # bbox_targets是检测框的四条边和它对应的prior的偏移量，要换成原图上的框，和gt进行回归损失计算
            bbox_targets[pos_inds, :] = pos_gt_bboxes

            # TODO：训练时，这里报了一个错，需要解决： Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
            # 怀疑前面的zeros是在cpu里的，zeros_like 仿照定义，在gpu里了
            pts_targets[pos_inds, :] = pos_gt_points                                     # TODO: 新增points_targets

            dist_targets[pos_inds, :] = (
                bbox2distance(center_priors[pos_inds, :2], pos_gt_bboxes)
                / center_priors[pos_inds, None, 2]
            )
            dist_targets = dist_targets.clamp(min=0, max=self.reg_max - 0.1)
            
            # 上面计算回归，这里就是得到用于计算类别损失的，把那些匹配到的prior利用pos_inds索引筛出来
            labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            label_scores[pos_inds] = pos_ious
            label_weights[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        return (
            labels,
            label_scores,
            label_weights,
            bbox_targets,
            pts_targets,                                                         # TODO: 新增pts_targets输出部分
            dist_targets,
            num_pos_per_img,
        )

    # 分配还是采用的是bbox的iou来进行标签的分配，然后顺带输出points
    def sample(self, assign_result, gt_bboxes, gt_points):          # TODO 新增gt_points部分
        """Sample positive and negative bboxes."""
        # 分配到正样本的priors索引
        pos_inds = (
            torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        # 负样本的priors的索引
        neg_inds = (
            torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
            pos_gt_points = torch.empty_like(gt_points).view(-1, 8)             # TODO   新增部分，新增 pos_gt_points
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
                gt_points = gt_points.view(-1, 8)
            # pos_gt_bboxes 大小为[正样本数,4] 4就是框的位置
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
            # pos_gt_points 大小为[正样本数,8] 8就是四个点的坐标
            pos_gt_points = gt_points[pos_assigned_gt_inds, :]
        return pos_inds, neg_inds, pos_gt_bboxes, pos_gt_points, pos_assigned_gt_inds   # TODO  新增输出 pos_gt_points

    def post_process(self, preds, meta):
        """Prediction results post processing. Decode bboxes and rescale
        to original image size.
        Args:
            preds (Tensor): Prediction output.
            meta (dict): Meta info.
        """
        cls_scores, bbox_preds = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )
        result_list = self.get_bboxes(cls_scores, bbox_preds, meta)
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
            det_bboxes, det_labels = result
            det_bboxes = det_bboxes.detach().cpu().numpy()
            det_bboxes[:, :4] = warp_boxes(
                det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height
            )
            classes = det_labels.detach().cpu().numpy()
            for i in range(self.num_classes):
                inds = classes == i
                det_result[i] = np.concatenate(
                    [
                        det_bboxes[inds, :4].astype(np.float32),
                        det_bboxes[inds, 4:5].astype(np.float32),
                    ],
                    axis=1,
                ).tolist()
            det_results[img_id] = det_result
        return det_results

    def show_result(
        self, img, dets, class_names, score_thres=0.3, show=True, save_path=None
    ):
        result = overlay_bbox_cv(img, dets, class_names, score_thresh=score_thres)
        if show:
            cv2.imshow("det", result)
        return result

    def get_bboxes(self, cls_preds, reg_preds, img_metas):
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
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
        scores = cls_preds.sigmoid()
        result_list = []
        for i in range(b):
            # add a dummy background class at the end of all labels
            # same with mmdetection2.0
            score, bbox = scores[i], bboxes[i]
            padding = score.new_zeros(score.shape[0], 1)
            score = torch.cat([score, padding], dim=1)
            results = multiclass_nms(
                bbox,
                score,
                score_thr=0.05,
                nms_cfg=dict(type="nms", iou_threshold=0.6),
                max_num=100,
            )
            result_list.append(results)
        return result_list

    # 在feature map上布置一组prior
    # prior就是框分布的回归起点，将以prior的位置作为目标中心，预测四个值形成检验框
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
        # feature map 的大小    40x40,20x20,10x10,5x5
        h, w = featmap_size 

        # arange 会生成一个一维的tensor，和range差不多，步长默认为1     feature map 和 stride 是对应的
        x_range = (torch.arange(w, dtype=dtype, device=device)) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device)) * stride
        
        # 根据网格长宽生成一组二维坐标
        y, x = torch.meshgrid(y_range, x_range)

        # 展平成一维
        y = y.flatten()
        x = x.flatten()

        # 扩充出一个strides的tensor，稍后给每一个prior都加上其对应的下采样倍数
        strides = x.new_full((x.shape[0],), stride)

        # 将得到的prior按照一下顺序叠成二维tensor，即原图上的坐标，采样倍数
        proiors = torch.stack([x, y, strides, strides], dim=-1)

        # 一次处理一个batch，所以unsqueeze增加一个batch的维度
        # 然后把得到的prior赋值到同个batch的其他位置上
        return proiors.unsqueeze(0).repeat(batch_size, 1, 1)


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
            cls_pred, reg_pred = output.split(
                [self.num_classes, 4 * (self.reg_max + 1)], dim=1
            )
            cls_pred = cls_pred.sigmoid()
            out = torch.cat([cls_pred, reg_pred], dim=1)
            outputs.append(out.flatten(start_dim=2))
        return torch.cat(outputs, dim=2).permute(0, 2, 1)

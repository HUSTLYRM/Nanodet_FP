import torch
import torch.nn.functional as F

from ...loss.iou_loss import bbox_overlaps
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

"""
    根据 pred 和 GT 的 IoU 进行软标签分配
    不添加 keypoints 的 oks 的代价了, 直接使用 bbox+cls 标签分配的结果
    
    YOLOX-Pose 虽然补充了oks的代价, 但是默认权重为0, 配置中并未赋值, 未使用
    参考: https://github.com/open-mmlab/mmpose/blob/main/projects/yolox_pose/models/assigner.py
"""


class DynamicSoftLabelAssigner(BaseAssigner):
    """Computes matching between predictions and ground truth with
    dynamic soft label assignment.
        使用动态软标签分配计算预测与真实值之间的匹配
    Args:
        topk (int): Select top-k predictions to calculate dynamic k
            best matchs for each gt. Default 13.
            为每个gt选择k个最佳预测来计算动态k最佳匹配。默认值为13。
        iou_factor (float): The scale factor of iou cost. Default 3.0.
            IoU成本的缩放因子。默认值为3.0。
        ignore_iof_thr (int): whether ignore max overlaps or not.
            Default -1 (1 or -1).
            是否忽略最大重叠。
    """

    # nanodet-plus head 中 topk设置为 13, 其余均为默认值
    def __init__(self, topk=13, iou_factor=3.0, ignore_iof_thr=-1):
        self.topk = topk
        self.iou_factor = iou_factor
        self.ignore_iof_thr = ignore_iof_thr

    def assign(
        self,
        pred_scores,
        priors,
        decoded_bboxes,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
    ):
        """Assign gt to priors with dynamic soft label assignment.
            使用动态软标签分配将gt分配给先验
        Args:
            pred_scores (Tensor): Classification scores of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
                类别得分
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
                所有先验点, 先验中心以及对应步长
            decoded_bboxes (Tensor): Predicted bboxes, a 2D-Tensor with shape
                [num_priors, 4] in [tl_x, tl_y, br_x, br_y] format.
                预测 bboxes
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
                真实 bboxes
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
                真实 labels
        Returns:
            :obj:`AssignResult`: The assigned result.
            标签分配结果
        """
        INF = 100000000
        num_gt = gt_bboxes.size(0)
        num_bboxes = decoded_bboxes.size(0)

        # assign 0 by default
        # 与 decoded_bboxes 在同一设备上, 长度为 num_bboxes, 类型为 torch.long 的一维向量
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)

        # 锚点中心 (N, 2)
        prior_center = priors[:, :2]
        # (N, M, 2) <= (N, 1, 2) - (M, 2)   广播规则
        lt_ = prior_center[:, None] - gt_bboxes[:, :2]
        # 同上
        rb_ = gt_bboxes[:, 2:] - prior_center[:, None]

        # 合并左上角和右下角的相对位置信息 (N, M, 4)
        deltas = torch.cat([lt_, rb_], dim=-1)
        # 判断 N个 锚点是否在 M个gt_bboxes 内部, 得到 (N, M) 尺寸的向量
        is_in_gts = deltas.min(dim=-1).values > 0
        # (N, M) => (N, ), 对每个锚点, 判断是否有一个 gt_bboxes 包含它
        # valid_mask 表示了用于预测的锚点
        valid_mask = is_in_gts.sum(dim=1) > 0

        # 只保留 valid_mask 为 True 的锚点
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        # 只保留 valid_mask 为 True 的预测得分
        valid_pred_scores = pred_scores[valid_mask]
        # 有效的数量
        num_valid = valid_decoded_bbox.size(0)

        # 如果没有 gt_bboxes, 没有预测框或者没有有效匹配, 则直接返回空的分配结果
        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = decoded_bboxes.new_full(
                    (num_bboxes,), -1, dtype=torch.long
                )
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels
            )

        # 计算有效匹配的预测框与 gt_bboxes 之间的 IoU
        pairwise_ious = bbox_overlaps(valid_decoded_bbox, gt_bboxes)
        # 计算 IoU 的损失
        iou_cost = -torch.log(pairwise_ious + 1e-7)

        # 将真实的类别转换成 onehot 编码
        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1])
            .float()
            .unsqueeze(0)
            .repeat(num_valid, 1, 1)
        )
        # 赋值有效类别的分数
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)

        # 生成软标签, 考虑 IoU 权重
        soft_label = gt_onehot_label * pairwise_ious[..., None]
        scale_factor = soft_label - valid_pred_scores.sigmoid()

        # 使用二元交叉熵损失计算分类损失
        cls_cost = F.binary_cross_entropy_with_logits(
            valid_pred_scores, soft_label, reduction="none"
        ) * scale_factor.abs().pow(2.0)

        cls_cost = cls_cost.sum(dim=-1)

        # 计算总代价
        cost_matrix = cls_cost + iou_cost * self.iou_factor

        # 进行动态 K-matching, 得到匹配的部分锚点, 这些锚点每个都对应一个 gt_bbox
        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(
            cost_matrix, pairwise_ious, num_gt, valid_mask
        )

        # convert to AssignResult format
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        # 得到分配的类别
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        
        # 最大 iou
        max_overlaps = assigned_gt_inds.new_full(
            (num_bboxes,), -INF, dtype=torch.float32
        )
        # 锚点对应的最大 iou
        max_overlaps[valid_mask] = matched_pred_ious

        if (
            self.ignore_iof_thr > 0                     # false
            and gt_bboxes_ignore is not None
            and gt_bboxes_ignore.numel() > 0
            and num_bboxes > 0
        ):
            ignore_overlaps = bbox_overlaps(
                valid_decoded_bbox, gt_bboxes_ignore, mode="iof"
            )
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            assigned_gt_inds[ignore_idxs] = -1

        # 真实标注的个数, priors 分配的 gt 索引, 对应的 iou, 分配的标签
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels
        )

    # 根据预测框与真实框之间 IoU 以及损失矩阵来进行匹配
    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        """Use sum of topk pred iou as dynamic k. Refer from OTA and YOLOX.

        Args:
            cost (Tensor): Cost matrix.
            pairwise_ious (Tensor): Pairwise iou matrix.
            num_gt (int): Number of gt.
            valid_mask (Tensor): Mask for valid bboxes.
        """
        # 初始化一个与 cost 同形状的匹配矩阵
        matching_matrix = torch.zeros_like(cost)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.topk, pairwise_ious.size(0))
        # 选取每个真实框的前 k 个最高 IoU 值     (candidate_topk, num_gt)
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        # 计算每个 gt 的前k个最高 IoU 值的和, 作为动态 k, 即这里的 k 会根据 topk_iou 变化   (num_gt)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        # 进行动态匹配, 遍历 gt_bboxes
        for gt_idx in range(num_gt):
            # 选取当前 gt_bbox 对应的损失矩阵中前 k 个最小损失值的索引, 即对应的锚点的索引
            # cost 维度为 (num_priors, num_gt)
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            # 将对应的匹配矩阵中的值置为 1, 对应的 dynamic_k 个锚点被选中 
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        # matching_matrix 尺寸为 (num_priors, num_gt), 挑选出匹配的锚点    (num_priors)
        # 一个锚点与两个或更多个 gt_bbox 匹配
        prior_match_gt_mask = matching_matrix.sum(1) > 1

        # 如果一个锚点和多个 gt_bboxes 匹配, 那么则选择代价最小的那一个
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0.0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0

        # 更新有效匹配的标志
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0.0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        # 获取有效匹配的每个预测框对应的 gt_bbox 的索引 maching_matrix     (num_priors, num_gts)
        # argmax 获取最大的那一个值的索引       (num_valid_priors)
        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        # 计算有效匹配的每个预测框与 gt_bbox 之间的 IoU     (num_valid_priors)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]
        # 每个 prior 与一个gt_bbox对应 (一个gt_bbox可以对应多个预测框) 
        return matched_pred_ious, matched_gt_inds
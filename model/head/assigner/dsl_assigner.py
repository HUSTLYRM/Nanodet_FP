import torch
import torch.nn.functional as F

from ...loss.iou_loss import bbox_overlaps
from ...loss.wing_loss import points_loss
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

# 这部分参考 https://github.com/1248289414/nanodet_keypoint/blob/rmdet/nanodet/model/head/assigner/dsl_assigner.py 修改
# dynamic soft label assigner 根据pred和gt的iou进行软标签分配，某个pred与gt的iou越大，最终分配给它的标签值会接近一，反之会变小
class DynamicSoftLabelAssigner(BaseAssigner):
    """Computes matching between predictions and ground truth with
    dynamic soft label assignment.

    Args:
        topk (int): Select top-k predictions to calculate dynamic k
            best matchs for each gt. Default 13.
        iou_factor (float): The scale factor of iou cost. Default 3.0.
        ignore_iof_thr (int): whether ignore max overlaps or not.
            Default -1 (1 or -1).
    """

    def __init__(self, topk=13, iou_factor=3.0, ignore_iof_thr=-1):
        self.topk = topk
        self.iou_factor = iou_factor
        self.ignore_iof_thr = ignore_iof_thr

    def assign(                                         # TODO 修改标签分配策略
        self,
        pred_scores,
        priors,
        decoded_bboxes,
        decoded_points,     # TODO
        gt_bboxes,
        gt_points,          # TODO
        gt_labels,
        gt_bboxes_ignore=None,
    ):
        """Assign gt to priors with dynamic soft label assignment.
        Args:
            pred_scores (Tensor): Classification scores of one image,               分类得分2Dtensor
                a 2D-Tensor with shape [num_priors, num_classes]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape        一个图片的锚点
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Predicted bboxes, a 2D-Tensor with shape       预测的检测框 和 以上的锚点是对应的
                [num_priors, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor       一个图片各个检测框的实际标注
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are       
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor          一个图片中各个识别框实际标注类别
                with shape [num_gts].

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        INF = 100000000
        num_gt = gt_bboxes.size(0)                                      # 实际标注的数量
        num_bboxes = decoded_bboxes.size(0)                             # 检验框的数量

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)

        # 切片得到prior的为位置（类似anchor point的中心点）
        prior_center = priors[:, :2]        # [num_priors, 2]

        # 计算prior center到GT左上角和右下角的距离，从而判断prior是否在GT框内, 得到的是  每个prior center和每个 gt的 距离
        lt_ = prior_center[:, None] - gt_bboxes[:, :2]                  # [:, None]在不改变数据的情况下，追加一个维度
        rb_ = gt_bboxes[:, 2:] - prior_center[:, None]

        deltas = torch.cat([lt_, rb_], dim=-1)
        # is_in_gts 通过判断deltas全部大于0，筛选出处在gt中的prior 锚点
        # [dxlt,dylt,dxrb,dyrb] 四个值都需要大于零，则他们中的最小值也要大于0
        is_in_gts = deltas.min(dim=-1).values > 0
        # 这一步生成有效的prior的索引，这里注意之所以使用sum是因为一个prior可能落在多个gt中
        # 因此上一步生成的is_in_gts确定的是某个prior是否落在每一个gt中，只要落在一个gt范围内，便是有效的
        valid_mask = is_in_gts.sum(dim=1) > 0

        # 利用得到的mask确定由哪些prior生成的pred_box和它们对应的scores是有效的
        # valid_decoded_bbox和valid_pred_scores的长度是落在gt中prior的个数
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_decoded_points = decoded_points[valid_mask]          
        valid_pred_scores = pred_scores[valid_mask]

        num_valid = valid_decoded_bbox.size(0)

        # 如果没有预测框或者训练样本中没有gt的情况， TODO 图片没有label时可能会报错
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
        
        # 计算bbox和gt的iou损失
        pairwise_ious = bbox_overlaps(valid_decoded_bbox, gt_bboxes)
        # 加上一个很小的数防止出现NaN
        iou_cost = -torch.log(pairwise_ious + 1e-7)

        pts_cost = points_loss(valid_decoded_points, gt_points)             # TODO
        
        # 根据num_valid的数量（有效bbox）生成对应长度的one-hot label之后用于计算soft lable
        # 每个匹配到gt的prior都会有一个tensor, label位置的元素为1，其余为0 （one-hot编码）
        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1])
            .float()
            .unsqueeze(0)
            .repeat(num_valid, 1, 1)
        )
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)

        # one-hot * IOU得到软标签， 预测框和gt越接近，预测越好
        soft_label = gt_onehot_label * pairwise_ious[..., None]
        # 差距越大，说明当前的预测效果越差，稍后的cost计算应该给一个更大的惩罚
        scale_factor = soft_label - valid_pred_scores

        # 计算分类交叉熵损失
        cls_cost = F.binary_cross_entropy(
            valid_pred_scores, soft_label, reduction="none"
        ) * scale_factor.abs().pow(2.0)

        cls_cost = cls_cost.sum(dim=-1)
 
        # 得到匹配开销矩阵，数值为分类损失，iou损失，这里利用iou_factor作为调制系数
        cost_matrix = cls_cost + iou_cost * self.iou_factor + pts_cost          # TODO

        # 返回值为分配到标签的prior与它们对应的gt的iou和这些prior匹配到的gt的索引
        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(
            cost_matrix, pairwise_ious, num_gt, valid_mask
        )

        # convert to AssignResult format
        # 把结果还原为priors的长度
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full(
            (num_bboxes,), -INF, dtype=torch.float32
        )
        max_overlaps[valid_mask] = matched_pred_ious

        if (
            self.ignore_iof_thr > 0
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

        return AssignResult(    
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels
        )

    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        """Use sum of topk pred iou as dynamic k. Refer from OTA and YOLOX.

        Args:
            cost (Tensor): Cost matrix.
            pairwise_ious (Tensor): Pairwise iou matrix.
            num_gt (int): Number of gt.
            valid_mask (Tensor): Mask for valid bboxes.
        """
        matching_matrix = torch.zeros_like(cost)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0.0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0.0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds

import torch
from torchvision.ops import nms


def multiclass_nms(
    multi_bboxes, multi_keypoints, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None, num_keypoints=4
):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
        keypoints = multi_keypoints.view(multi_scores.size(0), -1, num_keypoints * 2)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)
        keypoints = multi_keypoints[:, None].expand(multi_scores.size(0), num_classes, num_keypoints * 2)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    # 表示哪些得分超过阈值的 bboxes
    valid_mask = scores > score_thr

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # we have to use this ugly code
    # 选出得分超过阈值的 bboxes
    bboxes = torch.masked_select(
        bboxes, torch.stack((valid_mask, valid_mask, valid_mask, valid_mask), -1)
    ).view(-1, 4)
    
    # TODO: 为了适应不同关键点, 应该再次进行修改
    # 同样的, 选出得分超过阈值的 keypoints
    keypoints = torch.masked_select(
        keypoints, torch.stack((valid_mask, valid_mask, valid_mask, valid_mask, valid_mask, valid_mask, valid_mask, valid_mask), -1)
    ).view(-1, num_keypoints * 2)

    if score_factors is not None:
        scores = scores * score_factors[:, None]
    
    # 选出得分超过阈值的 scores
    scores = torch.masked_select(scores, valid_mask)
    # 类别索引  valid_mask.nonzero(as_tuple=False) 得到的第一列为样本索引, 第二列为类别索引, 这里选用了第二列
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if bboxes.numel() == 0:
        outputs = multi_bboxes.new_zeros((0, 4 + 2 * num_keypoints + 1))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError(
                "[ONNX Error] Can not record NMS "
                "as it has not been executed this time"
            )
        return outputs, labels

    # 进行 nms, dets 包括了 bboxes, keypoints, scores
    dets, keep = batched_nms(bboxes, keypoints, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]

# bboxes, scores, idxs 均为与处理过的, 得分超过阈值的部分
# idxs 为类别的索引
def batched_nms(boxes, keypoints, scores, idxs, nms_cfg, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.
    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.
    Arguments:
        boxes (torch.Tensor): boxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict): specify nms type and other parameters like iou_thr.
            Possible keys includes the following.
            - iou_thr (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
                number of boxes is large (e.g., 200k). To avoid OOM during
                training, the users could set `split_thr` to a small value.
                If the number of boxes is greater than the threshold, it will
                perform NMS on each group of boxes separately and sequentially.
                Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class.
    Returns:
        tuple: kept dets and indice.
    """
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop("class_agnostic", class_agnostic)
    
    if class_agnostic:                              # 不区分类别, 进行 nms
        boxes_for_nms = boxes
    else:                                           # 区分类别, 让不同类别不会影响
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]

    nms_cfg_.pop("type", "nms")
    split_thr = nms_cfg_.pop("split_thr", 10000)
    
    # 如果要进行nms的框的数量小于split_thr, 则直接进行nms, 否则划分一下
    if len(boxes_for_nms) < split_thr:
        # 候选框, 得分 进行nms
        keep = nms(boxes_for_nms, scores, **nms_cfg_)
        # 选出保留的框
        boxes = boxes[keep]
        keypoints = keypoints[keep]             # TODO: 选择keypoints
        scores = scores[keep]
    else:
        # 记录一个全为0的mask
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        # 对于每个类别的索引
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            keep = nms(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True
        keep = total_mask.nonzero(as_tuple=False).view(-1)
        keep = keep[scores[keep].argsort(descending=True)]
        # 选出保留的框
        boxes = boxes[keep]
        keypoints = keypoints[keep]             # TODO: 选择keypoints
        scores = scores[keep]

    return torch.cat([boxes, keypoints, scores[:, None]], -1), keep

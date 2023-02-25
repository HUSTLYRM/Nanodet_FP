import torch
from torchvision.ops import nms


def multiclass_nms(                                                                                 # TODO: 新增multi_pts
    multi_bboxes, multi_scores, multi_pts, score_thr, nms_cfg, max_num=-1, score_factors=None
):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column                 # 最后增加一个，表示背景类别
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
    
    # print("---nms--multiclass_nms---")
    # 类别数
    num_classes = multi_scores.size(1) - 1  # 36， 减去背景的类别

    # print(multi_scores.shape)       # [2125, 36+1] num_classes + padding
    # print(multi_bboxes.shape)       # [2125, 4]
    # print(multi_pts.shape)          # [2125, 8]

    # exclude background category
    if multi_bboxes.shape[1] > 4:   
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
        points = multi_pts.view(multi_scores.size(0), -1, 8)                            # TODO 仿照bboxes增加points
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)     # [:None], 在数据不便的情况下，增加一个维度   expand进行扩展，并且expand只能对维度值等于1的那个维度进行扩展
        points = multi_pts[:, None].expand(multi_scores.size(0), num_classes, 8)        # TODO 同上
    scores = multi_scores[:, :-1]
    
    # print(scores.shape)       # [2125, 36]
    # print(bboxes.shape)       # [2125, 36, 4]         增加成为了36类别
    # print(points.shape)       # [2125, 36, 8]

    # filter out boxes with low scores  过滤掉得分很低的部分, 筛选出大于阈值的那些目标
    valid_mask = scores > score_thr     # 相当于一个bool的列表  [2125, 36] 可以认为是 一个bool张量

    # We use masked_select for ONNX exporting purpose,      
    # which is equivalent to bboxes = bboxes[valid_mask]
    # we have to use this ugly code                         # 找出2125xnum_classes个目标中, 得分比较高的一些结果，并将结果分割成[num, 4]大小
    bboxes = torch.masked_select(
        bboxes, torch.stack((valid_mask, valid_mask, valid_mask, valid_mask), -1)
    ).view(-1, 4)

    # print(bboxes.shape)       # [num, 4]

    points = torch.masked_select(                           # 找出2125xnum_classes个目标中, 得分比较高的一些结果，并将结果分割成[num, 4]大小 # TODO 仿照bboxes增加
        points, torch.stack((valid_mask, valid_mask, valid_mask, valid_mask, valid_mask, valid_mask, valid_mask, valid_mask), -1)
    ).view(-1, 8)

    # print(points.shape)       # [num, 8]

    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)        # torch.Size([0])
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]       # torch.Size([0])

    if bboxes.numel() == 0:                 # numel()函数是统计张量的个数，总共有多少个数据 eg [1,2,3] ,则返回 1x2x3=6， 6个数据
        bboxes_pts = multi_bboxes.new_zeros((0, 5))                 # torch.Size([0, 5])        points的结果也存放在bboxes里面了，方便列表后续的解析（不过单独写出来也不错）
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)     # torch.Size([0])
        points = multi_pts.new_zeros((0, 8))                      # torch.Size([0, 9])               # TODO 仿照bboxes增加points的输出

        # print(bboxes.shape)
        # print(points.shape)
        # print(labels.shape)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError(
                "[ONNX Error] Can not record NMS "
                "as it has not been executed this time"
            )
        return bboxes_pts, labels, points                                              # TODO 增加返回points, 注意返回的顺序： bboxes, points, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    # print(dets.shape)
    # print(keep.shape)

    # 只取出max_num个结果, 只选出最高的几个
    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep], points[keep]


def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):          
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
        tuple: kept dets and indice.        # indice 指标,标记体
    """
    
    # print("---nms--batch_nms---")
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop("class_agnostic", class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        # print("max_coordinate")
        # print(max_coordinate)
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
    nms_cfg_.pop("type", "nms")
    split_thr = nms_cfg_.pop("split_thr", 10000)
    if len(boxes_for_nms) < split_thr:
        keep = nms(boxes_for_nms, scores, **nms_cfg_)   
        boxes = boxes[keep]
        scores = scores[keep]                             
    else:
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            keep = nms(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True

        keep = total_mask.nonzero(as_tuple=False).view(-1)
        keep = keep[scores[keep].argsort(descending=True)]
        boxes = boxes[keep]
        scores = scores[keep]

    # print(points.shape)
    # print(boxes.shape)
    # print(scores.shape)
    # print(keep)
    return torch.cat([boxes, scores[:, None]], -1), keep    #TODO 修改 det(4, 8, num_class) ，keep (keep类似于一种选择)
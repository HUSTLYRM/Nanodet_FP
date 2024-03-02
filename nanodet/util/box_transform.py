import torch

# 将积分后的距离与锚点结合转换成真实的 bbox
def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)

# TODO: 仿照distance2bbox新增distance2pts
# 这里仅针对四点排序为:
# 1  4
# 2  3
# 这一点限制了前面的发展
# 所以要注意
def distance2keypoints(points, distance, max_shape=None):
    # 左上点
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    # 左下点
    x2 = points[..., 0] - distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    # 右下点
    x3 = points[..., 0] + distance[..., 4]
    y3 = points[..., 1] + distance[..., 5]
    # 右上点
    x4 = points[..., 0] + distance[..., 6]
    y4 = points[..., 1] - distance[..., 7]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
        x3 = x3.clamp(min=0, max=max_shape[1])
        y3 = y3.clamp(min=0, max=max_shape[0])
        x4 = x4.clamp(min=0, max=max_shape[1])
        y4 = y4.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2, x3, y3, x4, y4], -1)


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        bbox (Tensor): Shape (n, 4), "xyxy" format
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=

    Returns:
        Tensor: Decoded distances.
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)

# TODO: 个人觉得和上面的部分对应即可, 但是有待验证, 目前只适用于
# 1  4 
# 2  3
# 这样排布的四关键点模型
def keypoints2distance(center, keypoints, max_dis=None, eps=0.1):
    # 左上点
    x1 = center[:, 0] - keypoints[:, 0]
    y1 = center[:, 1] - keypoints[:, 1]
    # 左下点
    x2 = center[:, 0] - keypoints[:, 2]
    y2 = keypoints[:, 3] - center[:, 1]
    # 右下点
    x3 = keypoints[:, 4] - center[:, 0]
    y3 = keypoints[:, 5] - center[:, 1]
    # 右上点
    x4 = keypoints[:, 6] - center[:, 0]
    y4 = center[:, 1] - keypoints[:, 7]
    if max_dis is not None:
        x1 = x1.clamp(min=0, max=max_dis - eps)
        y1 = y1.clamp(min=0, max=max_dis - eps)
        x2 = x2.clamp(min=0, max=max_dis - eps)
        y2 = y2.clamp(min=0, max=max_dis - eps)
        x3 = x3.clamp(min=0, max=max_dis - eps)
        y3 = y3.clamp(min=0, max=max_dis - eps)
        x4 = x4.clamp(min=0, max=max_dis - eps)
        y4 = y4.clamp(min=0, max=max_dis - eps)
    return torch.stack([x1, y1, x2, y2, x3, y3, x4, y4], -1)
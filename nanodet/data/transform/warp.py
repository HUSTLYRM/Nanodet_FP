# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def get_flip_matrix(prob=0.5):
    F = np.eye(3)
    if random.random() < prob:
        F[0, 0] = -1
    return F


def get_perspective_matrix(perspective=0.0):
    """

    :param perspective:
    :return:
    """
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)
    return P


def get_rotation_matrix(degree=0.0):
    """

    :param degree:
    :return:
    """
    R = np.eye(3)
    a = random.uniform(-degree, degree)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=1)
    return R


def get_scale_matrix(ratio=(1, 1)):
    """

    :param ratio:
    """
    Scl = np.eye(3)
    scale = random.uniform(*ratio)
    Scl[0, 0] *= scale
    Scl[1, 1] *= scale
    return Scl


def get_stretch_matrix(width_ratio=(1, 1), height_ratio=(1, 1)):
    """

    :param width_ratio:
    :param height_ratio:
    """
    Str = np.eye(3)
    Str[0, 0] *= random.uniform(*width_ratio)
    Str[1, 1] *= random.uniform(*height_ratio)
    return Str


def get_shear_matrix(degree):
    """

    :param degree:
    :return:
    """
    Sh = np.eye(3)
    Sh[0, 1] = math.tan(
        random.uniform(-degree, degree) * math.pi / 180
    )  # x shear (deg)
    Sh[1, 0] = math.tan(
        random.uniform(-degree, degree) * math.pi / 180
    )  # y shear (deg)
    return Sh


def get_translate_matrix(translate, width, height):
    """

    :param translate:
    :return:
    """
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation
    return T


def get_resize_matrix(raw_shape, dst_shape, keep_ratio):
    """
    Get resize matrix for resizing raw img to input size
    :param raw_shape: (width, height) of raw image
    :param dst_shape: (width, height) of input image
    :param keep_ratio: whether keep original ratio
    :return: 3x3 Matrix
    """
    r_w, r_h = raw_shape
    d_w, d_h = dst_shape
    Rs = np.eye(3)
    if keep_ratio:
        C = np.eye(3)
        C[0, 2] = -r_w / 2
        C[1, 2] = -r_h / 2

        if r_w / r_h < d_w / d_h:
            ratio = d_h / r_h
        else:
            ratio = d_w / r_w
        Rs[0, 0] *= ratio
        Rs[1, 1] *= ratio

        T = np.eye(3)
        T[0, 2] = 0.5 * d_w
        T[1, 2] = 0.5 * d_h
        return T @ Rs @ C
    else:
        Rs[0, 0] *= d_w / r_w
        Rs[1, 1] *= d_h / r_h
        return Rs


def warp_and_resize(
    meta: Dict,
    warp_kwargs: Dict,
    dst_shape: Tuple[int, int],
    keep_ratio: bool = True,
):
    # TODO: background, type
    raw_img = meta["img"]
    height = raw_img.shape[0]  # shape(h,w,c)
    width = raw_img.shape[1]

    # center
    C = np.eye(3)
    C[0, 2] = -width / 2
    C[1, 2] = -height / 2

    # do not change the order of mat mul
    if "perspective" in warp_kwargs and random.randint(0, 1):
        P = get_perspective_matrix(warp_kwargs["perspective"])
        C = P @ C
    if "scale" in warp_kwargs and random.randint(0, 1):
        Scl = get_scale_matrix(warp_kwargs["scale"])
        C = Scl @ C
    if "stretch" in warp_kwargs and random.randint(0, 1):
        Str = get_stretch_matrix(*warp_kwargs["stretch"])
        C = Str @ C
    if "rotation" in warp_kwargs and random.randint(0, 1):
        R = get_rotation_matrix(warp_kwargs["rotation"])
        C = R @ C
    if "shear" in warp_kwargs and random.randint(0, 1):
        Sh = get_shear_matrix(warp_kwargs["shear"])
        C = Sh @ C
    if "flip" in warp_kwargs:
        F = get_flip_matrix(warp_kwargs["flip"])
        C = F @ C
    if "translate" in warp_kwargs and random.randint(0, 1):
        T = get_translate_matrix(warp_kwargs["translate"], width, height)
    else:
        T = get_translate_matrix(0, width, height)
    M = T @ C
    # M = T @ Sh @ R @ Str @ P @ C
    ResizeM = get_resize_matrix((width, height), dst_shape, keep_ratio)
    M = ResizeM @ M
    img = cv2.warpPerspective(raw_img, M, dsize=tuple(dst_shape))
    meta["img"] = img
    meta["warp_matrix"] = M
    if "gt_bboxes" in meta:
        boxes = meta["gt_bboxes"]
        meta["gt_bboxes"] = warp_boxes(boxes, M, dst_shape[0], dst_shape[1])
    if "gt_bboxes_ignore" in meta:
        bboxes_ignore = meta["gt_bboxes_ignore"]
        meta["gt_bboxes_ignore"] = warp_boxes(
            bboxes_ignore, M, dst_shape[0], dst_shape[1]
        )
    if "gt_masks" in meta:
        for i, mask in enumerate(meta["gt_masks"]):
            meta["gt_masks"][i] = cv2.warpPerspective(mask, M, dsize=tuple(dst_shape))

    # TODO: keypoints
    if "gt_keypoints" in meta:
        keypoints = meta["gt_keypoints"]
        meta["gt_keypoints"] = warp_keypoints(keypoints, M, dst_shape[0], dst_shape[1])

    return meta

# boxes 为 (n,4) 大小数组, n个框的 x1,y1,x2,y2
# 重新生成对应的 bbox
def warp_boxes(boxes, M, width, height):
    # 输入的框的数量
    n = len(boxes)
    if n:
        # warp points
        # 初始化 (4n,3) 大小的数组, 并填充为 1
        xy = np.ones((n * 4, 3))
        # 重新排列输入边界框的坐标，并存储在 xy 的前两列
        # xy 为 4n 行, 3 列
        # (x1, y1, 1)
        # (x2, y2, 1)
        # (x1, y2, 1)
        # (x2, y1, 1)
        # 这样组合是得到 bbox 的四个角点的坐标, 将四个点进行映射
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        # 坐标乘以变换矩阵
        xy = xy @ M.T  # transform
        # xy 现在为 4n 行, 2 列, 归一化后的内容, 包含映射后的 bbox 的四个点
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        # create new boxes
        # 创建新的边界框, 从变换后的点钟提取 x 和 y 坐标
        # x 为 bbox 映射后四个点的 x 坐标
        # y 为 bbox 映射后四个点的 y 坐标
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        # 统计每个框的 x y 坐标的最小值和最大值, 合并成数组, 新的 bbox 的左上点和右下点
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        # clip boxes
        # 裁剪边界框, 保证在 0-width, 0-height 范围
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        # 返回新的 bbox
        return xy.astype(np.float32)
    else:
        return boxes


# def warp_keypoints(keypoints, M, width, height):
#     n = len(keypoints)
#     if n:
#         # warp points
#         xy = np.ones((n * 4, 3))
#         # x1y1, x2y2, x1y2, x2y1
#         xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
#         xy = xy @ M.T  # transform
#         xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
#         # create new boxes
#         x = xy[:, [0, 2, 4, 6]]
#         y = xy[:, [1, 3, 5, 7]]
#         xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
#         # clip boxes
#         xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
#         xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
#         return xy

# keypoints (n, num*2), 为 n 个关键点的xy坐标（在数据读取的时候(yolo格式), 就将可见性去除了）
# 矫正过的映射, 借助 num_keypoints, 使得不仅适用于四点模型
def warp_keypoints(keypoints, M, width, height):
    # 图片中标注的个数
    n = len(keypoints)
    if n:
        # 获取关键点的个数, 其中 // 3 的 3 表示 x,y,vis
        num_keypoints = len(keypoints[0]) // 3
        # warp points
        xy = np.ones((n * num_keypoints, 3))
        # 组织 x1y1, x2y2 ..., 首先 reshape
        # 然后 reshape 成为 n * num_keypoints 行, 3 列的数组
        xy[:, :3] = keypoints.reshape(num_keypoints * n, 3)
        # 赋值为1, 用于后续矩阵运算
        xy[:, 2] = 1
        # 对坐标进行变换
        xy = xy @ M.T  # transform
        # 归一, 留下第 3 列, 恰好作为可见性的
        xy = (xy[:, :3] / xy[:, 2:3]).reshape(n, num_keypoints * 3)  # rescale, 正好最后一列归一为1, 也能表示可见性
        # clip
        # 控制在 0-width, 0-height
        xy[:, range(0, 3 * num_keypoints, 3)] = xy[:, range(0, 3 * num_keypoints, 3)].clip(0, width)
        xy[:, range(1, 3 * num_keypoints, 3)] = xy[:, range(1, 3 * num_keypoints, 3)].clip(0, height)
        # 关键点映射后的坐标 (xy, xy)
        return xy
    else:
        return keypoints

# 对于网络的输出进行 warp
def warp_keypoints_2d(keypoints, M, width, height):
    # 图片中标注的个数
    n = len(keypoints)
    if n:
        # 获取关键点的个数, 其中 // 2 的 2 表示 x,y
        num_keypoints = len(keypoints[0]) // 2
        # warp points
        xy = np.ones((n * num_keypoints, 3))
        # 组织 x1y1, x2y2 ..., 首先 reshape
        # 然后 reshape 成为 n * num_keypoints 行, 3 列的数组
        xy[:, :2] = keypoints.reshape(num_keypoints * n, 2)
        # 对坐标进行变换
        xy = xy @ M.T  # transform
        # 归一, 留下第 3 列, 恰好作为可见性的
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, num_keypoints * 2)  # rescale, 正好最后一列归一为1, 也能表示可见性
        # clip
        # 控制在 0-width, 0-height
        xy[:, range(0, 2 * num_keypoints, 2)] = xy[:, range(0, 2 * num_keypoints, 2)].clip(0, width)
        xy[:, range(1, 2 * num_keypoints, 2)] = xy[:, range(1, 2 * num_keypoints, 2)].clip(0, height)
        # 关键点映射后的坐标 (xy, xy)
        return xy
    else:
        return keypoints

def get_minimum_dst_shape(
    src_shape: Tuple[int, int],
    dst_shape: Tuple[int, int],
    divisible: Optional[int] = None,
) -> Tuple[int, int]:
    """Calculate minimum dst shape"""
    src_w, src_h = src_shape
    dst_w, dst_h = dst_shape

    if src_w / src_h < dst_w / dst_h:
        ratio = dst_h / src_h
    else:
        ratio = dst_w / src_w

    dst_w = int(ratio * src_w)
    dst_h = int(ratio * src_h)

    if divisible and divisible > 0:
        dst_w = max(divisible, int((dst_w + divisible - 1) // divisible * divisible))
        dst_h = max(divisible, int((dst_h + divisible - 1) // divisible * divisible))
    return dst_w, dst_h


class ShapeTransform:
    """Shape transforms including resize, random perspective, random scale,
    random stretch, random rotation, random shear, random translate,
    and random flip.

    Args:
        keep_ratio: Whether to keep aspect ratio of the image.
        divisible: Make image height and width is divisible by a number.
        perspective: Random perspective factor.
        scale: Random scale ratio.
        stretch: Width and height stretch ratio range.
        rotation: Random rotate degree.
        shear: Random shear degree.
        translate: Random translate ratio.
        flip: Random flip probability.
    """

    def __init__(
        self,
        keep_ratio: bool,
        divisible: int = 0,
        perspective: float = 0.0,
        scale: Tuple[int, int] = (1, 1),
        stretch: Tuple = ((1, 1), (1, 1)),
        rotation: float = 0.0,
        shear: float = 0.0,
        translate: float = 0.0,
        flip: float = 0.0,
        **kwargs
    ):
        self.keep_ratio = keep_ratio
        self.divisible = divisible
        self.perspective = perspective
        self.scale_ratio = scale
        self.stretch_ratio = stretch
        self.rotation_degree = rotation
        self.shear_degree = shear
        self.flip_prob = flip
        self.translate_ratio = translate

    # TODO: 数据增强的类
    def __call__(self, meta_data, dst_shape):
        # 获取原始图像
        raw_img = meta_data["img"]
        height = raw_img.shape[0]  # shape(h,w,c)
        width = raw_img.shape[1]

        # center    初始化一个仿射变换的中心
        C = np.eye(3)
        C[0, 2] = -width / 2
        C[1, 2] = -height / 2

        # 获取透视变换矩阵 P
        P = get_perspective_matrix(self.perspective)
        C = P @ C

        # 获取尺度变换矩阵
        Scl = get_scale_matrix(self.scale_ratio)
        C = Scl @ C

        # 获取拉伸变换矩阵
        Str = get_stretch_matrix(*self.stretch_ratio)
        C = Str @ C

        # 获取旋转变换矩阵
        R = get_rotation_matrix(self.rotation_degree)
        C = R @ C

        # 获取剪切变换矩阵
        Sh = get_shear_matrix(self.shear_degree)
        C = Sh @ C

        # 获取翻转变换矩阵
        F = get_flip_matrix(self.flip_prob)
        C = F @ C

        # 获取平移变换矩阵
        T = get_translate_matrix(self.translate_ratio, width, height)
        M = T @ C

        # 保持宽高比
        if self.keep_ratio:
            dst_shape = get_minimum_dst_shape(
                (width, height), dst_shape, self.divisible
            )

        # 获取缩变换矩阵 ResizeM
        ResizeM = get_resize_matrix((width, height), dst_shape, self.keep_ratio)
        M = ResizeM @ M

        # 原始图片映射得到 img, 其中映射矩阵为 M
        img = cv2.warpPerspective(raw_img, M, dsize=tuple(dst_shape))
        meta_data["img"] = img
        meta_data["warp_matrix"] = M
        
        # 除了对图片进行映射, 对相应的标注也要进行相应的映射, 包括 bbox 以及 keypoints
        if "gt_bboxes" in meta_data:
            boxes = meta_data["gt_bboxes"]
            meta_data["gt_bboxes"] = warp_boxes(boxes, M, dst_shape[0], dst_shape[1])
        if "gt_bboxes_ignore" in meta_data:
            bboxes_ignore = meta_data["gt_bboxes_ignore"]
            meta_data["gt_bboxes_ignore"] = warp_boxes(
                bboxes_ignore, M, dst_shape[0], dst_shape[1]
            )
        if "gt_masks" in meta_data:
            for i, mask in enumerate(meta_data["gt_masks"]):
                meta_data["gt_masks"][i] = cv2.warpPerspective(
                    mask, M, dsize=tuple(dst_shape)
                )

        # TODO: 添加对于 ketpoints 的映射（针对增强的部分）
        if "gt_keypoints" in meta_data:
            keypoints = meta_data["gt_keypoints"]
            meta_data["gt_keypoints"] = warp_keypoints(keypoints, M, dst_shape[0], dst_shape[1])

        return meta_data

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from shapely.geometry import Polygon, MultiPoint, polygon # 多边形相关


#
#  TODO 参考 TUP-NN-Train 新增的多边形iou计算（以下实现中就是采用的4点）
#  后续可以考虑进行 使用
# 
class PolyIOUloss(nn.Module):
    """
        For Caculating iou loss between two polygons
    """

    def __init__(self, reduction="none", loss_type="iou", loss_weight=1.0):             # 默认类型是iou, 只有两个类型： iou或者giou
        super(PolyIOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type
        self.loss_weight = loss_weight                                                  # 给定一个loss_weight, 方便在yml中给定参数

    def forward(self, preds, targets):                                                  # [x1,y1,x2,y2,x3,y3,x4,y4], 的预测和实际
        ious = []
        if self.loss_type == "giou":
            gious = []

        for pred, target in zip(preds, targets):                        # TODO 随后选择使用一下这个损失函数时，再注意下是否需要这个zip (暂时保留，个人人为不需要）
            pred = pred.reshape(4, 2)       # 预测值                    [x1,y1,x2,y2,x3,y3,x4,y4]
            target = target.reshape(4, 2)   # 实际值

            pred_poly = Polygon(pred).convex_hull                       # 预测四边形
            target_poly = Polygon(target).convex_hull                   # 实际四边形
            union_poly = Polygon(torch.cat((pred, target))).convex_hull # 并集 U 部分

            if self.loss_type == "iou":
                if not pred_poly.intersects(target_poly):               # 不相交
                    iou = 0
                else:
                    iou = pred_poly.intersection(target_poly).area / union_poly.area    # 计算交并比
                ious.append(iou)

            elif self.loss_type == "giou":                              # 如果采用的是giou
                if not pred_poly.intersects(target_poly):               # 不相交
                    iou = 0
                    giou = -1
                else:
                    iou = pred_poly.intersection(target_poly).area / union_poly.area    # 交并比
                    giou = iou - \
                        ((union_poly.area - pred_poly.intersection(target_poly).area) / union_poly.area)
                    # 计算giou

                ious.append(iou)
                gious.append(giou)


        ious = torch.tensor(ious)
        if self.loss_type == "giou":
            gious = torch.tensor(gious)

        if self.loss_type == "iou":
            loss = 1 - ious
        elif self.loss_type == "giou":
            loss = 1 - gious
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return self.loss_weight*loss                                                    # TODO 新增 self.loss_weight
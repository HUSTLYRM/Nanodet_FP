import torch
import torch.nn as nn
import math
from .utils import weighted_loss


# TODO 新增关键点检测的 wingloss损失函数
class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2, loss_weight=1.0):       # TODO 新增加了loss_weight，为了和其他iou损失函数保持一致
        super(WingLoss, self).__init__()
        self.omega = omega                                                              # 给定w     omega
        self.epsilon = epsilon                                                          # 给定e     epsilon
        self.loss_weight = loss_weight                                                  # TODO 新增部分
        self.C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)      # 计算C
        
    def forward(self, pred, target):                                    # pred和target 都是 list
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()                                     # 其实： nanodet网络 输出 的是一个 [x1,y1,x2,y2,x3,y3,x4,y4] 的list, 相当于横纵坐标分别求loss，再求和
        delta_y1 = delta_y[delta_y < self.omega]                        # |x| < w 的部分 
        delta_y2 = delta_y[delta_y >= self.omega]                       # |x| >= w 的部分
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        loss2 = delta_y2 - self.C
        return self.loss_weight * ((loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2)))     # TODO 增加了 self.loss_weight 用来方便在yml文件中设置权重

# points_loss这部分来自： https://github.com/1248289414/nanodet_keypoint/blob/rmdet/nanodet/model/loss/wing_loss.py
def points_loss(pre_x, gt_x, alpha=1.03, beta=100):
    cost = torch.sub(pre_x[...,None,:],gt_x[...,None,:,:]).abs().sum(-1) / gt_x.size(-1)
    cost = torch.pow(alpha, cost - beta)
    return cost
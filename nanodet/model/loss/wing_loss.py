import torch
import torch.nn as nn
import math

# TODO: 添加 wing_loss
class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2, loss_weight=1.0):
        super(WingLoss, self).__init__()
        self.omega = omega                                                              # 给定w     omega
        self.epsilon = epsilon                                                          # 给定e     epsilon
        self.loss_weight = loss_weight
        self.C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)      # 计算C
        
    def forward(self, pred, target):                                    # pred和target 都是 list
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()                                     # 其实： nanodet网络 输出 的是一个 [x1,y1,x2,y2,x3,y3,x4,y4] 的list, 相当于横纵坐标分别求loss，再求和
        delta_y1 = delta_y[delta_y < self.omega]                        # |x| < w 的部分 
        delta_y2 = delta_y[delta_y >= self.omega]                       # |x| >= w 的部分
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        loss2 = delta_y2 - self.C
        return self.loss_weight * ((loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2)))
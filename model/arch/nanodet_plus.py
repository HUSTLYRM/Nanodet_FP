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

import copy

import torch

from ..head import build_head
from .one_stage_detector import OneStageDetector


class NanoDetPlus(OneStageDetector):
    def __init__(
        self,
        backbone,
        fpn,
        aux_head,
        head,
        detach_epoch=0,
    ):
        super(NanoDetPlus, self).__init__(
            backbone_cfg=backbone, fpn_cfg=fpn, head_cfg=head
        )
        self.aux_fpn = copy.deepcopy(self.fpn)
        self.aux_head = build_head(aux_head)
        self.detach_epoch = detach_epoch

    def forward_train(self, gt_meta):
        img = gt_meta["img"]                    # resize处理过的图片，输入网络的部分，gt_meta["raw_img"]是原始图片
        feat = self.backbone(img)               # 使用backbone处理
        fpn_feat = self.fpn(feat)               # 使用fpn处理
        
        """
            <下面这一部分代码不是很清楚>
            
            查到的一些资料如下：
            tensor.detach()返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置，不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad
            使用这个新的tensor进行计算式，后面进行反向传播时，到调用detach()的tensor时就会停止，不能再继续向前进行传播
            使用detach得到的tensor和原始的tensor共同一个内存，即一个修改另一个也会跟着改变

            个人的理解是：达到了对应的epoch后，部分进行了冻结，反向传播到这里就停止传播了
        """
        if self.epoch >= self.detach_epoch:     
            aux_fpn_feat = self.aux_fpn([f.detach() for f in feat])
            dual_fpn_feat = (
                torch.cat([f.detach(), aux_f], dim=1)
                for f, aux_f in zip(fpn_feat, aux_fpn_feat)
            )
        else:
            aux_fpn_feat = self.aux_fpn(feat)
            dual_fpn_feat = (
                torch.cat([f, aux_f], dim=1) for f, aux_f in zip(fpn_feat, aux_fpn_feat)
            )

        head_out = self.head(fpn_feat)          # nanodet_plus_head
        aux_head_out = self.aux_head(dual_fpn_feat) # 辅助训练头
        loss, loss_states = self.head.loss(head_out, gt_meta, aux_preds=aux_head_out)   # 调用head的loss，计算损失
        return head_out, loss, loss_states                                              # 返回预测结果 以及 损失

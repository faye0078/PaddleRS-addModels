# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Transferred from https://github.com/rcdaudt/fully_convolutional_change_detection/blob/master/siamunet_conc.py .

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .layers import Conv3x3, MaxPool2x2, ConvTransposed3x3, Identity
from .backbones.fcn import FCNHead
from .backbones.hrnet import HRNet_W48


class HRNet_FCN(nn.Layer):


    def __init__(self,
                 in_channels=3,
                 num_classes=2,
                 use_mixed_loss=False):
        super(HRNet_FCN, self).__init__()

        self.backbone = HRNet_W48(in_channels)
        backbone_indices = [0]
        backbone_channels = [
            self.backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head = FCNHead(
            num_classes,
            backbone_indices,
            backbone_channels,)


    def forward(self, x):
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)
        return [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear') for logit in logit_list
        ]


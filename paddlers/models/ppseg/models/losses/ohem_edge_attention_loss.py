# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import nn
import paddle.nn.functional as F

from paddlers.models.ppseg.cvlibs import manager
from paddlers.models.ppseg.models import losses


@manager.LOSSES.add_component
class OhemEdgeAttentionLoss(nn.Layer):
    """
    Implements the cross entropy loss function. It only compute the edge part.

    Args:
        edge_threshold (float, optional): The pixels greater edge_threshold as edges. Default: 0.8.
        thresh (float, optional): The threshold of ohem. Default: 0.7.
        min_kept (int, optional): The min number to keep in loss computation. Default: 5000.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self,
                 edge_threshold=0.8,
                 thresh=0.7,
                 min_kept=5000,
                 ignore_index=255):
        super().__init__()
        self.edge_threshold = edge_threshold
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        self.EPS = 1e-10

    def forward(self, logits, label):
        """
        Forward computation.

        Args:
            logits (tuple|list): (seg_logit, edge_logit) Tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1. C =1 of edge_logit .
            label (Tensor): Label tensor, the data type is int64. Shape is (N, C), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, C, D1, D2,..., Dk), k >= 1.
        """
        seg_logit, edge_logit = logits[0], logits[1]
        if len(label.shape) != len(seg_logit.shape):
            label = paddle.unsqueeze(label, 1)
        if edge_logit.shape != label.shape:
            raise ValueError(
                'The shape of edge_logit should equal to the label, but they are {} != {}'
                .format(edge_logit.shape, label.shape))

        # Filter out edge
        filler = paddle.ones_like(label) * self.ignore_index
        label = paddle.where(edge_logit > self.edge_threshold, label, filler)

        # ohem
        n, c, h, w = seg_logit.shape
        label = label.reshape((-1, ))
        valid_mask = (label != self.ignore_index).astype('int64')
        num_valid = valid_mask.sum()
        label = label * valid_mask

        prob = F.softmax(seg_logit, axis=1)
        prob = prob.transpose((1, 0, 2, 3)).reshape((c, -1))

        if self.min_kept < num_valid and num_valid > 0:
            # let the value which ignored greater than 1
            prob = prob + (1 - valid_mask)

            # get the prob of relevant label
            label_onehot = F.one_hot(label, c)
            label_onehot = label_onehot.transpose((1, 0))
            prob = prob * label_onehot
            prob = paddle.sum(prob, axis=0)

            threshold = self.thresh
            if self.min_kept > 0:
                index = prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                threshold_index = int(threshold_index.numpy()[0])
                if prob[threshold_index] > self.thresh:
                    threshold = prob[threshold_index]
                kept_mask = (prob < threshold).astype('int64')
                label = label * kept_mask
                valid_mask = valid_mask * kept_mask
        # make the invalid region as ignore
        label = label + (1 - valid_mask) * self.ignore_index
        label = label.reshape((n, 1, h, w))
        valid_mask = valid_mask.reshape((n, 1, h, w)).astype('float32')

        loss = F.softmax_with_cross_entropy(
            seg_logit, label, ignore_index=self.ignore_index, axis=1)
        loss = loss * valid_mask
        avg_loss = paddle.mean(loss) / (paddle.mean(valid_mask) + self.EPS)

        label.stop_gradient = True
        valid_mask.stop_gradient = True
        return avg_loss

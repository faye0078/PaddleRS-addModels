import numpy as np
import paddle.nn as nn
import paddle
from paddlers.models.ppseg.cvlibs import manager, param_init
from paddlers.models.ppseg.utils import utils

OPS = {
    "conv1x1": lambda C_in, C_out, stride, affine, repeats=1: nn.Sequential(
        conv1x1(C_in, C_out, stride=stride),
        nn.BatchNorm2D(C_out),
        nn.ReLU(),
    ),
    "conv3x3": lambda C_in, C_out, stride, affine, repeats=1: nn.Sequential(
        conv3x3(C_in, C_out, stride=stride),
        nn.BatchNorm2D(C_out),
        nn.ReLU(),
    ),
    "conv3x3_dil3": lambda C_in, C_out, stride, affine, repeats=1: nn.Sequential(
        conv3x3(C_in, C_out, stride=stride, dilation=3),
        nn.BatchNorm2D(C_out),
        nn.ReLU(),
    ),
    "conv3x3_dil12": lambda C_in, C_out, stride, affine, repeats=1: nn.Sequential(
        conv3x3(C_in, C_out, stride=stride, dilation=12),
        nn.BatchNorm2D(C_out),
        nn.ReLU(),
    ),
    "sep_conv_3x3": lambda C_in, C_out, stride, affine, repeats=1: SepConv(
        C_in, C_out, 3, stride, 1, affine=affine, repeats=repeats
    ),
    "sep_conv_5x5": lambda C_in, C_out, stride, affine, repeats=1: SepConv(
        C_in, C_out, 5, stride, 2, affine=affine, repeats=repeats
    ),
    "sep_conv_7x7": lambda C_in, C_out, stride, affine, repeats=1: SepConv(
        C_in, C_out, 7, stride, 3, affine=affine, repeats=repeats
    ),
    "avg_pool_3x3": lambda C_in, C_out, stride, affine, repeats=1: Pool(
        C_in, C_out, stride, repeats, ksize=3, mode="avg"
    ),
    "max_pool_3x3": lambda C_in, C_out, stride, affine, repeats=1: Pool(
        C_in, C_out, stride, repeats, ksize=3, mode="max"
    ),
    "global_average_pool": lambda C_in, C_out, stride, affine, repeats=1: GAPConv1x1(
        C_in, C_out
    )
}

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2D(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation
    )


def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution"
    return nn.Conv2D(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0
    )


def conv_bn(C_in, C_out, kernel_size, stride, padding, affine=True):
    return nn.Sequential(
        nn.Conv2D(C_in, C_out, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2D(C_out),
    )


def conv_bn_relu(C_in, C_out, kernel_size, stride, padding, affine=True):
    return nn.Sequential(
        nn.Conv2D(C_in, C_out, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2D(C_out),
        nn.ReLU(),
    )


def conv_bn_relu6(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2D(inp, oup, 3, stride, 1),
        nn.BatchNorm2D(oup),
        nn.ReLU6(),
    )


def conv_1x1_bn_relu6(inp, oup):
    return nn.Sequential(
        nn.Conv2D(inp, oup, 1, 1, 0),
        nn.BatchNorm2D(oup),
        nn.ReLU6(),
    )


class InvertedResidual(nn.Layer):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2D(inp, inp * expand_ratio, 1, 1, 0),
            nn.BatchNorm2D(inp * expand_ratio),
            nn.ReLU6(),
            # dw
            nn.Conv2D(
                inp * expand_ratio,
                inp * expand_ratio,
                3,
                stride,
                1,
                groups=inp * expand_ratio
            ),
            nn.BatchNorm2D(inp * expand_ratio),
            nn.ReLU6(),
            # pw-linear
            nn.Conv2D(inp * expand_ratio, oup, 1, 1, 0),
            nn.BatchNorm2D(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class Pool(nn.Layer):
    """Conv1x1 followed by pooling"""

    def __init__(self, C_in, C_out, stride, repeats, ksize, mode):
        super(Pool, self).__init__()
        self.conv1x1 = conv_bn(C_in, C_out, 1, 1, 0)
        if mode == "avg":
            self.pool = nn.AvgPool2D(
                ksize, stride=stride, padding=(ksize // 2)
            )
        elif mode == "max":
            self.pool = nn.MaxPool2D(ksize, stride=stride, padding=(ksize // 2))
        else:
            raise ValueError("Unknown pooling method {}".format(mode))

    def forward(self, x):
        x = self.conv1x1(x)
        return self.pool(x)


class GAPConv1x1(nn.Layer):
    """Global Average Pooling + conv1x1"""

    def __init__(self, C_in, C_out):
        super(GAPConv1x1, self).__init__()
        self.conv1x1 = conv_bn_relu(C_in, C_out, 1, stride=1, padding=0)

    def forward(self, x):
        size = x.shape[2:]
        out = x.mean(2, keepdim=True).mean(3, keepdim=True)
        out = self.conv1x1(out)
        out = nn.functional.interpolate(
            out, size=size, mode="bilinear", align_corners=False
        )
        return out


class DilConv(nn.Layer):
    """Dilated separable convolution"""

    def __init__(
        self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True
    ):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv2D(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in
            ),
            nn.Conv2D(C_in, C_out, kernel_size=1, padding=0),
            nn.BatchNorm2D(C_out),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Layer):
    """Separable convolution"""

    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation=1,
        affine=True,
        repeats=1,
    ):
        super(SepConv, self).__init__()

        def basic_op(C_in, C_out):
            return nn.Sequential(
                nn.Conv2D(
                    C_in,
                    C_in,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=C_in
                ),
                nn.Conv2D(C_in, C_out, kernel_size=1, padding=0),
                nn.BatchNorm2D(C_out),
                nn.ReLU(),
            )

        self.op = nn.Sequential()
        for idx in range(repeats):
            if idx > 0:
                C_in = C_out
            self.op.add_sublayer("sep_{}".format(idx), basic_op(C_in, C_out))

    def forward(self, x):
        return self.op(x)

class MixedRetrainCell(nn.Layer):
    
    def __init__(self, C_in, C_out, arch):
        super(MixedRetrainCell, self).__init__()
        self.scale = 1
        self._ops = nn.LayerList()
        for i, op_name in enumerate(OPS):
            if arch[i] == 1:
                op = OPS[op_name](C_in, C_out, 1, True)
                self._ops.append(op)
        self.ops_num = len(self._ops)
        self.scale = C_in/C_out


    def forward(self, x):
        if self.scale != 0:
            feature_size_h = self.scale_dimension(x.shape[2], self.scale)
            feature_size_w = self.scale_dimension(x.shape[3], self.scale)
            x = nn.functional.interpolate(x, [feature_size_h, feature_size_w], mode='bilinear', align_corners=True)
        return sum(op(x) for op in self._ops)

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))


class DNAS(nn.Layer):

    def __init__(self, layers, depth, connections, cell_arch, num_classes, base_multiplier=40, pretrained=None):
        '''
        Args:
            layers: layer × depth： one or zero, one means true
            depth: the model scale depth
            connections: the node connections
            cell: cell type
            dataset: dataset
            base_multiplier: base scale multiplier
        '''
        super(DNAS, self).__init__()
        self.pretrained = pretrained
        self.block_multiplier = 1
        self.base_multiplier = base_multiplier
        self.depth = depth
        self.layers = layers
        self.connections = connections
        self.node_add_num = np.zeros([len(layers), self.depth])
        cell = MixedRetrainCell

        half_base = int(base_multiplier // 2)
        input_channel = 3
        self.stem0 = nn.Sequential(
            nn.Conv2D(input_channel, half_base * self.block_multiplier, 3, stride=2, padding=1),
            nn.BatchNorm2D(half_base * self.block_multiplier),
            nn.ReLU()
        )
        self.stem1 = nn.Sequential(
            nn.Conv2D(half_base * self.block_multiplier, half_base * self.block_multiplier, 3, stride=1, padding=1),
            nn.BatchNorm2D(half_base * self.block_multiplier),
            nn.ReLU()
        )
        self.stem2 = nn.Sequential(
            nn.Conv2D(half_base * self.block_multiplier, self.base_multiplier * self.block_multiplier, 3, stride=2,padding=1),
            nn.BatchNorm2D(self.base_multiplier * self.block_multiplier),
            nn.ReLU()
        )
        self.cells = nn.LayerList()
        multi_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        num_last_features = 0
        for i in range(len(self.layers)):
            self.cells.append(nn.LayerList())
            for j in range(self.depth):
                self.cells[i].append(nn.LayerDict())
                num_connect = 0
                for connection in self.connections:
                    if ([i, j] == connection[1]).all():
                        num_connect += 1
                        if connection[0][0] == -1:
                            self.cells[i][j][str(connection[0])] = cell(self.base_multiplier * multi_dict[0],
                                                         self.base_multiplier * multi_dict[connection[1][1]], cell_arch[i][j])
                        else:
                            self.cells[i][j][str(connection[0])] = cell(self.base_multiplier * multi_dict[connection[0][1]],
                                                self.base_multiplier * multi_dict[connection[1][1]], cell_arch[i][j])
                self.node_add_num[i][j] = num_connect

                if i == len(self.layers) -1 and num_connect != 0:
                    num_last_features += self.base_multiplier * multi_dict[j]

        self.last_conv = nn.Sequential(nn.Conv2D(num_last_features, 256, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2D(256),
                                       nn.Dropout(0.5),
                                       nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2D(256),
                                       nn.Dropout(0.1),
                                       nn.Conv2D(256, num_classes, kernel_size=1, stride=1))
        
        self.init_weights()


        print('connections number: \n' + str(self.node_add_num))

    def forward(self, x):
        features = []

        temp = self.stem0(x)
        temp = self.stem1(temp)
        pre_feature = self.stem2(temp)

        for i in range(len(self.layers)):
            features.append([])
            for j in range(self.depth):
                features[i].append(0)
                k = 0
                for connection in self.connections:
                    if ([i, j] == connection[1]).all():
                        if connection[0][0] == -1:
                            features[i][j] += self.cells[i][j][str(connection[0])](pre_feature)
                        else:
                            if isinstance(features[connection[0][0]][connection[0][1]], int):
                                continue
                            features[i][j] += self.cells[i][j][str(connection[0])](features[connection[0][0]][connection[0][1]])
                        k += 1

        last_features = [feature for feature in features[len(self.layers)-1] if paddle.is_tensor(feature)]
        last_features = [nn.Upsample(size=last_features[0].shape[2:], mode='bilinear', align_corners=True)(feature) for feature in last_features]
        result = paddle.concat(last_features, axis=1)
        result = self.last_conv(result)
        result = nn.Upsample(size=x.shape[2:], mode='bilinear', align_corners=True)(result)
        logit_list = []
        logit_list.append(result)
        return logit_list

    def init_weights(self):
            for layer in self.sublayers():
                if isinstance(layer, nn.Conv2D):
                    param_init.normal_init(layer.weight, std=0.001)
                elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                    param_init.constant_init(layer.weight, value=1.0)
                    param_init.constant_init(layer.bias, value=0.0)
            if self.pretrained is not None:
                utils.load_pretrained_model(self, self.pretrained)
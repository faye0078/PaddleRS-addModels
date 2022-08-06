from paddlers.models.ppseg.models import DNAS
import paddle
import numpy as np
layers = np.ones([14, 4])
cell_arch = np.load(
                '/media/dell/DATA/wy/Seg_NAS/model/model_encode/GID-5/14layers_mixedcell1_3operation/cell_operations_0.npy')
connections = np.load(
                '/media/dell/DATA/wy/Seg_NAS/model/model_encode/GID-5/14layers_mixedcell1_3operation/third_connect_4.npy')
model = DNAS(layers, 4, connections, cell_arch, 5)

input1 = paddle.rand([2, 3, 224, 224])

output = model(input1)
a = 1


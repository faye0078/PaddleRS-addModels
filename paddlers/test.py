import sys
sys.path.append('..')
from custom_models.cd import HRNet_FCN
import paddle
model = HRNet_FCN(in_channels=6)

input1 = paddle.rand([2, 6, 224, 224])
# input2 = paddle.rand([2, 3, 224, 224])

output = model(input1)
a = 1


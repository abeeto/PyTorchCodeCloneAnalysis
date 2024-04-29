import torch
import torchvision
from torch import nn
from torch.nn import *
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2d(3, 3, 3, 1, 1)     # in_channel, out_channel, kernelsize, stride, padding
        self.maxpool = MaxPool2d(3, 3, ceil_mode=True)
        self.maxpool2 = MaxPool2d(3, ceil_mode=True)
        # kernel_size, stride, ceil_mode=True向上取整，不够保留，False-floor向下取整，不够舍弃

    def forward(self, input):
        output = self.conv1(input)
        return output

# 初始化网络
my_model = MyModel()

for data in dataloader:
    img, target = data
    img_out = my_model(img)
    print(img_out.shape)

# func_input = torch.tensor([[1, 2, 0, 3, 1],
#                            [0, 1, 2, 3, 1],
#                            [1, 2, 1, 0, 0],
#                            [5, 2, 3, 1, 1],
#                            [2, 1, 0, 1, 1]], dtype=torch.float32)
# kernel = torch.tensor([[1, 2, 1],
#                        [0, 1, 0],
#                        [2, 1, 0]])
# func_input = torch.reshape(func_input, (1, 1, 5, 5))      # batchsize(-1:自己计算batchsize), channel, h*w
# kernel = torch.reshape(kernel, (1, 1, 3, 3))
# -*- coding: utf-8 -*-
# Author: zero
# Time: 2022.07.16 15:48

import torchvision
from torch import nn
from torchvision.models import VGG16_Weights

vgg16 = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)  # 选择合适的weights
# vgg16下载的weights储存地址：/Users/sevenzero/.cache/torch/hub/checkpoints/vgg16-397923af.pth
print(vgg16)

train_data = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# vgg16.add_module('add_linear', nn.Linear(1000, 10))               # 将新模型添加到最后
# vgg16.classifier.add_module('add_linear', nn.Linear(1000, 10))      # 将新模型添加到制定部分
# vgg16.classifier[6] = nn.Linear(4096, 10)                           # 对制定模型层进行修改
print(vgg16)
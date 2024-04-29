# -*- coding: utf-8 -*-
# Author: zero
# Time: 2022.07.16 16:18
import torch

# 保存方式1 -> 加载方式1
import torchvision

model = torch.load("pth/vgg16_method1.pth")
# print(model)

# 保存方式2 -> 加载方式2
# 1 恢复模型结构
vgg16 = torchvision.models.vgg16()
# 2 加载模型参数（字典）
vgg16.load_state_dict(torch.load("pth/vgg16_method2.pth"))
print(vgg16)
# -*- coding: utf-8 -*-
# Author: zero
# Time: 2022.07.16 16:10
import torch
import torchvision
from torchvision.models import VGG16_Weights

vgg16 = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
# 保存方式1：模型结构+参数
torch.save(vgg16, "./pth/vgg16_method1.pth")

# 保存方式2：模型参数（官方推荐）
torch.save(vgg16.state_dict(), "./pth/vgg16_method2.pth")

# -*- coding: utf-8 -*-
# weibifan 2022-10-2, 2022-10-12
#  模型的加载，使用，保存  --- 类似于数据集的加载，使用，保存
# https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

'''
'''

import torch
import torchvision.models as models
from torchvision.models import VGG16_Weights

# 加载模型和指定参数
model = models.vgg16(weights=VGG16_Weights.DEFAULT)

# 保存参数
torch.save(model.state_dict(), '../model/model_weights.pth')

# 加载模型，没有参数
model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
# 加载指定的参数
model.load_state_dict(torch.load('model_weights.pth'))
# ？？？
model.eval()


#另外一种方法：
torch.save(model, '../model/model.pth')
model = torch.load('model.pth')
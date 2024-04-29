"""
Author : Avineil Jain 
This code defines the simple architecture used for the classification task. 
We use Transfer Learning, using the pre-trained ImageNet weights of ResNet50 and adding fc layers to it
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models 


class CNN(nn.Module):
	def __init__(self):
		super(CNN,self).__init__()
		self.base_model = models.resnet50(pretrained=True)
		self.resnet_fully_conv = nn.Sequential(*list(self.base_model.children())[:-2])
		self.avg_pool = nn.AvgPool2d(kernel_size=7,stride=1,padding=0)
		self.fc1 = nn.Linear(2048,256)
		self.dropout1 = nn.Dropout(p=0.4)
		self.fc2 = nn.Linear(256,8)

	def forward(self,x):
		x = self.resnet_fully_conv(x)
		x = self.avg_pool(x)
		x = x.view(x.size(0),-1)
		x = self.dropout1(self.fc1(x))
		out = self.fc2(x)

		return out 






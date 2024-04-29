import torch
import torch.nn as nn
import numpy as np 
import math
import time

def center_crop(layer,target_size):
	_,_,layer_width,layer_height = layer.size()
	start = (layer_width - target_size) // 2
	crop = layer[:,:,start:(Start+target_size),start:(start+target_size)]
	return crop

def concatenate(link,layer):
	crop = center_crop(link,layer.size()[2])
	concat = torch.cat([crop,layer],1)
	return concat

class unet(nn.Module):

	def __init__(self,n_class):
		super(unet,self).__init__()

		#Depth1
		self.conv1_1 = nn.Conv2d(3,64,3,padding=0)
		nn.init.xavier_uniform(self.conv1_1.weight)
		self.relu1_1 = nn.ReLU(inplace=True)
		self.bn1_1 = nn.BatchNorm2d(64)

		self.conv1_2 = nn.Conv2d(64,64,3,padding=0)
		nn.init.xavier_uniform(self.conv1_2.weight)
		self.relu1_2 = nn.ReLU(inplace=True)
		self.bn1_2 = nn.BatchNorm2d(64)

		#max pool 2
		self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

		#Depth2 
		self.conv2_1 = nn.Conv2d(64,128,3,padding=0)
		nn.init.xavier_uniform(self.conv2_1.weight)
		self.relu2_1 = nn.ReLU(inplace=True)
		self.bn2_1 = nn.BatchNorm2d(128)

		self.conv2_2 = nn.Conv2d(128,128,3,padding=0)
		nn.init.xavier_uniform(self.conv2_2.weight)
		self.relu2_2 = nn.ReLU(inplace=True)
		self.bn2_2 = nn.BatchNorm2d(128)

		#max pool 2
		self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

		#Depth3
		self.conv3_1 = nn.Conv2d(128,256,3,padding=0)
		nn.init.xavier_uniform(self.conv3_1.weight)
		self.relu3_1 = nn.ReLU(inplace=True)
		self.bn3_1 = nn.BatchNorm2d(256)

		self.conv3_2 = nn.Conv2d(256,256,3,padding=0)
		nn.init.xavier_uniform(self.conv3_2.weight)
		self.relu3_2 = nn.ReLU(inplace=True)
		self.bn3_2 = nn.BatchNorm2d(256)

		#max pool 3
		self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

		#Depth4
		self.conv4_1 = nn.Conv2d(256,512,3,padding=0)
		nn.init.xavier_uniform(self.conv4_1.weight)
		self.relu4_1 = nn.ReLU(inplace=True)
		self.bn4_1 = nn.BatchNorm2d(512)

		self.conv4_2 = nn.Conv2d(512,512,3,padding=0)
		nn.init.xavier_uniform(self.conv4_2.weight)
		self.relu4_2 = nn.ReLU(inplace=True)
		self.bn4_2 = nn.BatchNorm2d(512)

		#max pool 4
		self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

		#Depth5
		self.conv5_1 = nn.Conv2d(512,1024,3,padding=0)
		nn.init.xavier_uniform(self.conv5_1.weight)
		self.relu5_1 = nn.ReLU(inplace=True)
		self.bn5_1 = nn.BatchNorm2d()

		self.conv5_2 = nn.Conv2d(1024,1024,3,padding=0)
		nn.init.xavier_uniform(self.conv5_2.weight)
		self.relu5_2 = nn.ReLU(inplace=True)
		self.bn5_2 = nn.BatchNorm2d(1024)

		#upsample 4
		self.up4 = nn.ConvTranspose2d(1024,512,2,stride=2,bias=False)
		nn.init.xavier_uniform(self.up4.weight)

		#Depth4 up
		self.conv_up_4_1 = nn.Conv2d(1024,512,3,padding=0)
		nn.init.xavier_uniform(self.conv_up_4_1.weight)
		self.relu_up_4_1 = nn.ReLU(inplace=True)
		self.bn_up_4_1 = nn.BatchNorm2d(512)

		self.conv_up_4_2 = nn.Conv2d(512,512,3,padding=0)
		nn.init.xavier_uniform(self.conv_up_4_2.weight)
		self.relu_up_4_2 = nn.ReLU(inplace=True)
		self.bn_up_4_2 = nn.BatchNorm2d(512)

		#upsample 3
		self.up3 = nn.ConvTranspose2d(512,256,2,stride=2,bias=False)
		nn.init.xavier_uniform(self.up3.weight)

		#Depth3 up
		self.conv_up_3_1 = nn.Conv2d(512,256,3,padding=0)
		nn.init.xavier_uniform(self.conv_up_3_1.weight)
		self.relu_up_3_1 = nn.ReLU(inplace=True)
		self.bn_up_3_1 = nn.BatchNorm2d(256)

		self.conv_up_3_2 = nn.Conv2d(256,256,3,padding=0)
		nn.init.xavier_uniform(self.conv_up_3_2.weight)
		self.relu_up_3_2 = nn.ReLU(inplace=True)
		self.bn_up_3_2 = nn.BatchNorm2d(256)

		#upsample 2
		self.up2 = nn.ConvTranspose2d(256,128,2,stride=2,bias=False)
		nn.init.xavier_uniform(self.up2.weight)

		#Depth2 up
		self.conv_up_2_1 = nn.Conv2d(256,128,3,padding=0)
		nn.init.xavier_uniform(self.conv_up_2_1.weight)
		self.relu_up_2_1 = nn.ReLU(inplace=True)
		self.bn_up_2_1 = nn.BatchNorm2d()

		self.conv_up_2_2 = nn.Conv2d(128,128,3,padding=0)
		nn.init.xavier_uniform(self.conv_up_2_2.weight)
		self.relu_up_2_2 = nn.ReLU(inplace=True)
		self.bn_up_2_2 = nn.BatchNorm2d()

		#upsample 1
		self.up1 = nn.ConvTranspose2d(128,64,2,stride=2,bias=False)
		nn.init.xavier_uniform(self.up1.weight)

		#Depth1 up
		self.conv_up_1_1 = nn.Conv2d(128,64,3,padding=0)
		nn.init.xavier_uniform(self.conv_up_1_1.weight)
		self.relu_up_1_1 = nn.ReLU(inplace=True)
		self.bn_up_1_1 = nn.BatchNorm2d(64)

		self.conv_up_1_2 = nn.Conv2d(64,64,3,padding=0)
		nn.init.xavier_uniform(self.conv_up_1_2.weight)
		self.relu_up_1_2 = nn.ReLU(inplace=True)
		self.bn_up_1_2 = nn.BatchNorm2d(64)

		#Final output

		self.conv_score = nn.Conv2d(64,n_class,3,padding=0)
		nn.init.xavier_uniform(self.conv_score.weight)

	def forward(self,x):
		h = x
		h = self.bn1_1(self.relu1_1(self.conv1_1(h)))
		h = self.bn1_2(self.relu1_2(self.conv1_2(h)))
		link1 = h
		h = self.pool1(h)

		h = self.bn2_1(self.relu2_1(self.conv2_1(h)))
		h = self.bn2_2(self.relu2_2(self.conv2_2(h)))
		link2 = h
		h = self.pool2(h)

		h = self.bn3_1(self.relu3_1(self.conv3_1(h)))
		h = self.bn3_2(self.relu3_2(self.conv3_2(h)))
		link3 = h
		h = self.pool3(h)

		h = self.bn4_1(self.relu4_1(self.conv4_1(h)))
		h = self.bn4_2(self.relu4_2(self.conv4_2(h)))
		link4 = h
		h = self.pool4(h)

		h = self.bn5_1(self.relu5_1(self.conv5_1(h)))
		h = self.bn5_2(self.relu5_2(self.conv5_2(h)))

		h = self.up4(h)
		h = concatenate(link4,h)

		h = self.bn_up_4_1(self.relu_up_4_1(self.conv_up_4_1(h)))
		h = self.bn_up_4_2(self.relu_up_4_2(self.conv_up_4_2(h)))

		h = self.up3(h)
		h = concatenate(link3,h)

		h = self.bn_up_3_1(self.relu_up_3_1(self.conv_up_3_1(h)))
		h = self.bn_up_3_2(self.relu_up_3_2(self.conv_up_3_2(h)))

		h = self.up2(h)
		h = concatenate(link2,h)

		h = self.bn_up_2_1(self.relu_up_2_1(self.conv_up_2_1(h)))
		h = self.bn_up_2_2(self.relu_up_2_2(self.conv_up_2_2(h)))

		h = self.up1(h)
		h = concatenate(link1,h)

		h = self.bn_up_1_1(self.relu_up_1_1(self.conv_up_1_1(h)))
		h = self.bn_up_1_2(self.relu_up_1_2(self.conv_up_1_2(h)))

		h = self.conv_score(h)

		return h
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 21:31:40 2020

just a code to show off the author that even if he was walking in time series signals,
he is able to transfer the skills and knowledge to a slightly different problem in 
image processing/ 

in the following toy model, the assumption is that large and small images
requires 2 different pre process network that can be separatly trained
on the fly using torch's dynamic graph feature

"""
import torch
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import ImageFolder



train_dset = tf.keras.preprocessing.image_dataset_from_directory(r'./kaggledata/bee/bees/train')
test_dset = tf.keras.preprocessing.image_dataset_from_directory(r'./kaggledata/bee/bees/test')
validate_dset = tf.keras.preprocessing.image_dataset_from_directory(r'./kaggledata/bee/bees/validate')

train_dset2 = ImageFolder(r'./kaggledata/bee/bees/train')
test_dset2 = ImageFolder(r'./kaggledata/bee/bees/test')
validate_dset2 = ImageFolder(r'./kaggledata/bee/bees/validate')


small_cnt = 0
large_cnt = 0

import numpy as np
category_cnt = np.array([0]* 4)
# import numpy as np
if __name__ != '__main__':
    for img, label in train_dset2:
        category_cnt[label] += 1
        if np.all(np.array([128, 128]) > np.array(img).shape[:2]) :
            small_cnt += 1
        else:
            large_cnt += 1
    print(small_cnt, large_cnt)
    print(category_cnt)

class small_image_pre_process(nn.Module):
    # for images smaller than 256 by 256 pixelxs
    def __init__(self):
        super(small_image_pre_process, self).__init__()
        self.small_transform = torchvision.transforms.Compose([
                    torchvision.transforms.CenterCrop(64), 
                    torchvision.transforms.ToTensor()
                    ])
        self.conv1 = torch.nn.Conv2d(3, 10,  2 , 1)
        pass
    def forward(self, x):
        h = self.small_transform(x)
        h = torch.unsqueeze(h,0)
        h_relu = self.conv1(h).clamp(min=0)
        return h_relu
    pass



class large_image_pre_process(nn.Module):
    # for images that are 256 by 256 pixels or larger
    def __init__(self):
        super(large_image_pre_process, self).__init__()
        self.large_transform = torchvision.transforms.Compose([
                    torchvision.transforms.CenterCrop(128), 
                    torchvision.transforms.ToTensor()
                    ])
        self.conv1 = torch.nn.Conv2d(3, 10,  2 , 1, padding = (1,1))
        self.conv2 = torch.nn.Conv2d(10, 10,  2 , 1)
        pass
    def forward(self, x):
        h = self.large_transform(x)
        h = torch.unsqueeze(h,0)
        
        h_relu = self.conv1(h).clamp(min=0)
        h_relu = self.conv2(h_relu).clamp(min=0)
        return h_relu
    pass

def is_large(img):
    return np.all(np.array([128, 128]) < np.array(img).shape[:2])

def too_small(img):
    return np.all(np.array([64, 64]) > np.array(img).shape[:2])
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        # self.small_transform = torchvision.transforms.Compose([
        #             torchvision.transforms.CenterCrop(64), 
        #             torchvision.transforms.ToTensor()
        #             ])
        # self.large_transform = torchvision.transforms.Compose([
        #             torchvision.transforms.CenterCrop(128), 
        #             torchvision.transforms.ToTensor()
        #             ])
        self.small_image_pre_process = small_image_pre_process()
        self.large_image_pre_process = large_image_pre_process()
        self.conv1 = torch.nn.Conv2d(10, 10,  3 , 1)
        self.conv2 = torch.nn.Conv2d(10, 4,  3 , 1)
        self.avg_pool2d = torch.nn.AvgPool2d(32)
        
    
    def forward(self, x):
        
        if not is_large(x):
            # x = self.small_transform(x)
            # run max_pool stride 2 for 2 times to get 32 by 32 after cropping 128 by 128
            h = self.small_image_pre_process(x)
        else:
            # x = self.large_transform(x)
            # run max_pool stride 2 for 1 times to get 32 by 32 after cropping 64 by 64
            h = self.large_image_pre_process(x)
        
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.avg_pool2d(h)
        h = h.view(h.size(0), -1)
        return h
        # do later common upper layers after the preprocess layers
        
        
        
        pass
    pass
net = model()

print(net)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
loss_function = nn.NLLLoss()

n_epoch = 20
mini_batch_size = 16
mini_batch_cnt = 0
for i in range(n_epoch):
    net.zero_grad()
    # minibatch
    print('epoch', i)
    batch_num = 0
    for x, target in train_dset2:
        if too_small(x):
            continue
        train_out_probit = net(x)
        
        loss = loss_function(train_out_probit, torch.LongTensor([target]) ) 
        loss.backward()
        mini_batch_cnt += 1
        if mini_batch_cnt >= mini_batch_size:
            mini_batch_cnt = 0
            optimizer.step()
            net.zero_grad()
            batch_num += 1
            print('batch done', batch_num)
            pass
        pass
    mini_batch_cnt = 0
    optimizer.step()
    net.zero_grad()
    batch_num += 1
    print('batch done', batch_num)
    pass











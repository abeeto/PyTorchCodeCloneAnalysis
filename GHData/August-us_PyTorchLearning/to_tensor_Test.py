from __future__ import print_function
import torch
import torchvision.transforms as transforms
import numpy as np

# 定义转换方式，transforms.Compose将多个转换函数组合起来使用
transform1 = transforms.Compose([transforms.ToTensor()])  #归一化到(0,1)，简单直接除以255

# 定义一个数组
d1 = [1,2,3,4,5,6]
d2 = [4,5,6,7,8,9]
d3 = [7,8,9,10,11,14]
d4 = [11,12,13,14,15,15]
d5 = [d1,d2,d3,d4]
d = np.array([d5,d5,d5],dtype=np.float32)
d_t = np.transpose(d,(1,2,0)) # 转置为类似图像的shape，(H,W,C)，作为transform的输入
# 查看d的shape
print('d.shape: ',d.shape, '\n', 'd_t.shape: ', d_t.shape)


d_t_trans = transform1(d_t) # 直接使用函数归一化

# 手动归一化,下面的两个步骤可以在源码里面找到
d_t_temp = torch.from_numpy(d_t.transpose((2,0,1)))
d_t_trans_man = d_t_temp.float().div(255)
print (d_t_trans)
print(d_t_trans_man)

print(d_t_trans.equal(d_t_trans_man))
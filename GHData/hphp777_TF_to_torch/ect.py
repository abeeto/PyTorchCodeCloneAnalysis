import os, sys, pdb, time
from glob import glob
import pandas as pd
import numpy as np
import cv2
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from imageio import imread
from PIL import Image, ImageOps
import torchvision.models as models 
import torch.nn as nn
from matplotlib import pyplot as plt
import csv
import pandas as pd
import config
import shutil
import random

# Path, Classes(0 or 1)

# random.seed(1996)

# path = ['path']
# classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# for j in range(10):

#     all_image_paths = glob('C:/Users/hb/Desktop/data/CIFAR10_Client_random/C'+str(j)+'/*.png')
#     random.shuffle(all_image_paths) 

#     df = pd.DataFrame(columns=path + classes)    

#     for l in range(len(all_image_paths)):
#         label = [0] * 10
#         c = all_image_paths[l].split('/')[6].split('.')[0].split('_')[0].split('\\')[1]
#         i = classes.index(c)
#         label[i] = 1
#         row = []
#         row.append(all_image_paths[l])
#         row = row + label
#         df.loc[l] = row
    
#     df.to_csv('C:/Users/hb/Desktop/data/CIFAR10_Client_random/train'+str(j)+'.csv', index=False)

# all_image_paths = glob('C:/Users/hb/Desktop/data/CIFAR10_Client_random/test/*/*.png')
# random.shuffle(all_image_paths) 

# df = pd.DataFrame(columns=path + classes)    


# for l in range(len(all_image_paths)):
#     label = [0] * 10
#     c = all_image_paths[l].split('/')[6].split('\\')[2].split('_')[0]
#     i = classes.index(c)
#     label[i] = 1
#     row = []
#     row.append(all_image_paths[l])
#     row = row + label
#     df.loc[l] = row
    
# df.to_csv('C:/Users/hb/Desktop/data/CIFAR10_Client_random/test.csv', index=False)


# all_image_paths = glob('C:/Users/hb/Desktop/data/CIFAR10_Client_random/*/*.png')
# print(len(all_image_paths))

# path = 'C:/Users/hb/Desktop/data/CIFAR10_Client_random/train' + str(0) + '.csv'
# df = pd.read_csv(path, index_col=0)
# print(len(df))
# for i in range(1,10):
#     path = 'C:/Users/hb/Desktop/data/CIFAR10_Client_random/train' + str(i) + '.csv'
#     df = pd.concat([df, pd.read_csv(path, index_col=0)])
#     print(len(pd.read_csv(path, index_col=0)))

# df.to_csv('C:/Users/hb/Desktop/data/CIFAR10_Client_random/train.csv')

t1 = torch.Tensor([0,0,0,1])
t2 = torch.Tensor(4)

print(t2.view_as(t1))
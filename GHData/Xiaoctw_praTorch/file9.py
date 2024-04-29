from __future__ import print_function,division
import os
import torch
import pandas as pd
from skimage import io,transform
import numpy as np
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
#读取数据,将数据读入到（N,2）数组当中
#理解是这个csv文件相当于所有图片的配置文件
landmarks_frame=pd.read_csv('faces/face_landmarks.csv')
n=65
print(landmarks_frame.shape)
img_name=landmarks_frame.iloc[n,0]
landmarks=landmarks_frame.iloc[n,1:].as_matrix()#读入信息，标注点的信息
landmarks=landmarks.astype('float').reshape(-1,2)#进行转化
print("Image name:{}".format(img_name))
print("landmarks shape:{}".format(landmarks.shape))
print("First 4 Landmarks:{}".format(landmarks[:4]))

def show_landmarks(image,landmarks):
    plt.imshow(image)
    #画点
    plt.scatter(landmarks[:,0],landmarks[:,1],s=10,marker='.',c='r')
    plt.pause(0.001)

plt.figure()
show_landmarks(io.imread(os.path.join('faces/',img_name)),landmarks)
plt.show()
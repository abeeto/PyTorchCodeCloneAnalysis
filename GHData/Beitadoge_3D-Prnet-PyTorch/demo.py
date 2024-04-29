# -*- coding: utf-8 -*-

"""
    @date: 2019.07.18
    @author: samuel ko
    @func: PRNet Training Part.
"""
import os
import cv2
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.optim
from model.resfcn256 import ResFCN256

from tools.WLP300dataset import PRNetDataset, ToTensor, ToNormalize
from tools.prnet_loss import WeightMaskLoss, INFO

from utils.utils import save_image, test_data_preprocess, make_all_grids, make_grid
from utils.losses import SSIM

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils, models
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

import argparse
import glob
import ast

import scipy.io as sio
from skimage import io
import skimage.transform

#画关键点
end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype = np.int32) - 1
def plot_kpt(image, kpt):
    ''' Draw 68 key points
    Args: 
        image: the input image (image_w,image_h,3)
        kpt: (68, 3).
    '''
    image = image.copy()
    kpt = np.round(kpt).astype(np.int32)
    for i in range(kpt.shape[0]):
        st = kpt[i, :2]
        image = cv2.circle(image,(st[0], st[1]), 1, (0,0,255), 2)  
        if i in end_list:
            continue
        ed = kpt[i + 1, :2]
        image = cv2.line(image, (st[0], st[1]), (ed[0], ed[1]), (255, 255, 255), 1)
    return image

def get_landmarks(pos,tform=None):
    '''
    pos : 模型预测的UV Map, (256,256,3)
    tform : 原图 和 resize图 坐标转换矩阵
    '''

    if tform :
        #根据tform参数，将关键点映射回原坐标系中(未resize)
        cropped_vertices = np.reshape(pos, [-1, 3]).T #(3,65536)
        z = cropped_vertices[2,:].copy()/tform.params[0,0]
        cropped_vertices[2,:] = 1
        vertices = np.dot(np.linalg.inv(tform.params), cropped_vertices) #(3,65536)
        vertices = np.vstack((vertices[:2,:], z))
        pos = np.reshape(vertices.T, [256, 256, 3])


    uv_kpt_ind = np.loadtxt('./uv_kpt_ind.txt').astype(np.int32)
    kpt = pos[uv_kpt_ind[1,:],uv_kpt_ind[0,:], :]
    return kpt

def plot_keypoint(image,uv_map):
    '''
    image : ndarray (image_w,image_h,3)
    uv_map : ndarray (256,256,3)
    '''
    kpt = get_landmarks(uv_map)
    kpt_image = plot_kpt(image,kpt)
    return kpt_image

if __name__ == '__main__':
    image = cv2.imread('./Data/PRNet_PyTorch_Data/300WLP_IBUG/85766/original.jpg')[:,:,::-1]
    uv_map = np.load("./Data/PRNet_PyTorch_Data/300WLP_IBUG/85766/IBUG_image_043_1_3.npy")
    kpt_image = plot_keypoint(image,uv_map)
    cv2.imwrite("./1.jpg",kpt_image)


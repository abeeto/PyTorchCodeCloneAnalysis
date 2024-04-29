#%%
from cgi import test
from functools import total_ordering
import os
from unittest.mock import patch
import cv2
import random
import io

import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix ,roc_curve, auc, accuracy_score
from xgboost import XGBClassifier

import matplotlib.pyplot as plt 
from itertools import cycle
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms

# from model.patch_convmix_convnext import PatchConvmixConvnext
# from model.patch_RepLKNet_DRSN import PatchRepLKNetDRSN
from model.patch_convmix_Attention import PatchConvMixerAttention

from model.focal_loss import FocalLoss
import json

import pandas as pd
import seaborn as sns

import wandb
import time

import catboost as cb

from model.load_dataset import MyDataset, MultiEpochsDataLoader, CudaDataLoader
from model.assessment_tool import MyEstimator

import math
import logging

from argparse import ArgumentParser


SEED = 42
if SEED:
    '''設定隨機種子碼'''
    os.environ["PL_GLOBAL_SEED"] = str(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True




def cam_visualize(idx, model, _input, img_show, imgsavePath):
    #Image
    # print(img_show.shape)
    
    img = img_show[0].permute(1, 2, 0).to('cpu').detach().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.stack([img, img, img], axis=2)
    img = np.uint8(255 * img)
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(img)
    # plt.show()

    #HeatMap Mask
    pred, gap, test_feature = model(_input.to(device))

    # heatmap = torch.mean(gap, dim = 1).squeeze()
    heatmap = torch.clamp(gap, 0, 1)
    heatmap /= torch.max(heatmap)
    # print(heatmap.shape)
    heatmap = heatmap.permute(1,2,0)
    # print(heatmap.shape)

    heatmap = heatmap.to('cpu').detach().numpy()
    # plt.imshow(heatmap)
    # plt.show()

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
 
    heatmap = np.uint8(heatmap * 255)
    heatmapColor = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmapColor_remove = heatmapColor.copy()
    heatmapColor_remove[heatmap<60] = [0, 0, 0]
    # plt.imshow(heatmap)
    # plt.show()

    # print(np.min(heatmap), np.max(heatmap))

    dst = cv2.addWeighted(img, 0.5, heatmapColor_remove, 0.8, 1)
    cv2.imwrite(imgsavePath+'//'+ str(idx) + '_img.jpg', img)
    cv2.imwrite(imgsavePath+'//'+ str(idx) + '_heat.jpg', heatmapColor)
    cv2.imwrite(imgsavePath+'//'+ str(idx) + '_heatmap.jpg', dst)
    print("save as :", imgsavePath+'//'+ str(idx))
    return heatmapColor


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU State:', device)


    DATALOGPATH = r'C:\Data\surgical_temperature\trainingLogs\\'
    DATAPATH = r'C:\Data\surgical_temperature\color\\via_jet2_for_cam\\'

    train_mode = 0
    modelName = "(7d4G-5d3GB)_1GB_X4_CA_SA_RB_FL1_"

    logPath = DATALOGPATH + "//logs//" + str(modelName)
    imgsavePath = DATALOGPATH + "//cams//" + str(modelName)
    
    if not os.path.isdir(imgsavePath):
        os.mkdir(imgsavePath)

    Dataload = MyDataset(DATAPATH, DATALOGPATH, 2)
    dataset  = Dataload
    kf = KFold(n_splits = 10, shuffle = True)
    Kfold_cnt = 0

    for train_idx, val_idx in kf.split(dataset):
        Kfold_cnt += 1

        model = PatchConvMixerAttention(dim = 768, depth = 3, kernel_size = 9, patch_size = 16, n_classes = 2, train_mode = train_mode).to(device)

        saveModelpath = logPath + "//" + str(Kfold_cnt) + "_last.pth"
        model.eval()
        model.load_state_dict(torch.load(saveModelpath))
        model.to(device)


        train = Subset(dataset, train_idx)
        ML_train_loader = DataLoader(train, shuffle = False, num_workers = 1, persistent_workers = True)

        img_list = []
        # for idx, (x, y, key) in enumerate(ML_train_loader):
        #     img_list.append(x)

        for idx, (x, y, key) in enumerate(ML_train_loader):
            img_list.append(x)
            cam_visualize(idx, model, x.to(device), img_list[idx], imgsavePath)
            # break
        break





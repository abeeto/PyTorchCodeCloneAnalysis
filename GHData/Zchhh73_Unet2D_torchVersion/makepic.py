import os
import torch
import csv
import torchvision.utils as vutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision import datasets
from torch.autograd import Variable
from tensorboardX import SummaryWriter

UnetPath = 'D:\\train\\VerseData_UNet_Dice_withoutDS'
DeepResUnetPath = 'D:\\train\\VerseData_DeepResUNet_Dice_withoutDS'
DenseUnetPath = 'D:\\train\\VerseData_DenseUnet_DiceLoss_withoutDS'
DeepLabPath = 'D:\\train\\VerseData_DeepLab_DiceLoss_withoutDS'
DilatedResUnetPath = 'D:\\train\\VerseData_4L_DilatedResUnet_DiceLoss_withoutDS'

myfontdict = {'fontsize': 12, 'color': 'black'}


def loss_pic(path):
    file = pd.read_csv(path)
    title = path.split('\\')[-2].split('_')[-3]
    plt.title(title, fontdict=myfontdict)
    plt.plot(file['epoch'], file['loss'])
    plt.show()


def iou_pic(path):
    file = pd.read_csv(path)
    plt.plot(file['epoch'], file['iou'])
    plt.show()


def multi_loss_pic():
    unet_csv = os.path.join(UnetPath, 'log.csv')
    deepresunet_csv = os.path.join(DeepResUnetPath, 'log.csv')
    denseunet_csv = os.path.join(DenseUnetPath, 'log.csv')
    deeplab_csv = os.path.join(DeepLabPath, 'log.csv')
    dilatedresunet_csv = os.path.join(DilatedResUnetPath, 'log.csv')

    plt.title('loss', fontdict=myfontdict)
    unet_model = pd.read_csv(unet_csv, usecols=['epoch', 'loss'])
    plt.plot(unet_model.epoch, unet_model.loss, lw=1.5, label='UNet', color='blue')

    deepresunet_model = pd.read_csv(deepresunet_csv, usecols=['epoch', 'loss'])
    plt.plot(deepresunet_model.epoch, deepresunet_model.loss, lw=1.5, label='DeepResUnet', color='red')

    denseunet_model = pd.read_csv(denseunet_csv, usecols=['epoch', 'loss'])
    plt.plot(denseunet_model.epoch, denseunet_model.loss, lw=1.5, label='DenseUnet', color='black')

    deeplab_model = pd.read_csv(deeplab_csv, usecols=['epoch', 'loss'])
    plt.plot(deeplab_model.epoch, deeplab_model.loss, lw=1.5, label='DeepLab', color='pink')

    dilatedresunet_model = pd.read_csv(dilatedresunet_csv, usecols=['epoch', 'loss'])
    plt.plot(dilatedresunet_model.epoch, dilatedresunet_model.loss, lw=1.5, label='DilatedResUnet', color='green')
    plt.legend(loc=0)
    plt.show()


def multi_iou_pic():
    unet_csv = os.path.join(UnetPath, 'log.csv')
    deepresunet_csv = os.path.join(DeepResUnetPath, 'log.csv')
    denseunet_csv = os.path.join(DenseUnetPath, 'log.csv')
    deeplab_csv = os.path.join(DeepLabPath, 'log.csv')
    dilatedresunet_csv = os.path.join(DilatedResUnetPath, 'log.csv')

    plt.title('iou', fontdict=myfontdict)
    unet_model = pd.read_csv(unet_csv, usecols=['epoch', 'iou'])
    plt.plot(unet_model.epoch, unet_model.iou, lw=1.5, label='UNet', color='blue')

    deepresunet_model = pd.read_csv(deepresunet_csv, usecols=['epoch', 'iou'])
    plt.plot(deepresunet_model.epoch, deepresunet_model.iou, lw=1.5, label='DeepResUnet', color='red')

    denseunet_model = pd.read_csv(denseunet_csv, usecols=['epoch', 'iou'])
    plt.plot(denseunet_model.epoch, denseunet_model.iou, lw=1.5, label='DenseUnet', color='black')

    deeplab_model = pd.read_csv(deeplab_csv, usecols=['epoch', 'iou'])
    plt.plot(deeplab_model.epoch, deeplab_model.iou, lw=1.5, label='DeepLab', color='pink')

    dilatedresunet_model = pd.read_csv(dilatedresunet_csv, usecols=['epoch', 'iou'])
    plt.plot(dilatedresunet_model.epoch, dilatedresunet_model.iou, lw=1.5, label='DilatedResUnet', color='green')
    plt.legend(loc=0)
    plt.show()


if __name__ == '__main__':
    UnetModelCsv = os.path.join(UnetPath, 'log.csv')
    # loss_pic(UnetModelCsv)
    # iou_pic(UnetModelCsv)
    multi_loss_pic()
    multi_iou_pic()

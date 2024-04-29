import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from utils.dataset import VerseDataset

import model.ResUnet.model as ResUnetModel
import model.unet.unet_model as UnetModel

from metrics import dice_coef, batch_iou, mean_iou, iou_score, ppv, sensitivity
import losses
from utils.utils import str2bool, count_params
import joblib
from hausdorff import hausdorff_distance
import imageio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None,
                        help='model name')
    parser.add_argument('--mode', default=None,
                        help='GetPicture or Calculate')
    args = parser.parse_args()
    return args


def main():
    val_args = parse_args()
    args = joblib.load('trained_models/%s/args.pkl' % val_args.name)
    if not os.path.exists('output/%s' % args.name):
        os.makedirs('output/%s' % args.name)
    print('config ------')
    for arg in vars(args):
        print('%s %s' % (arg, getattr(args, arg)))
    print('-------------')
    joblib.dump(args, 'trained_models/%s/args.pkl' % args.name)

    # 创建模型
    print("=> creating model %s" % args.arch)
    model = ResUnetModel.__dict__[args.arch](args)
    model = model.cuda()

    # 数据集载入
    img_paths = glob(r'D:\data\test_data\img\*')
    mask_paths = glob(r'D:\data\test_data\mask\*')

    val_img_paths = img_paths
    val_mask_paths = mask_paths
    model.load_state_dict(torch.load('trained_models/%s/model.pth' % args.name))
    model.eval()

    val_dataset = VerseDataset(args, val_img_paths, val_mask_paths)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )

    if val_args.mode == "GetPicture":
        '''
        获取保存生成的标签图
        '''
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            with torch.no_grad():
                for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    input = input.cuda()
                    if args.deepsupervision:
                        output = model(input)[-1]
                    else:
                        output = model(input)
                    output = torch.sigmoid(output).data.cpu().numpy()
                    img_paths = val_img_paths[args.batch_size * i:args.batch_size * (i + 1)]

                    for i in range(output.shape[0]):
                        '''
                        生成灰色图片
                        '''
                        pass



if __name__ == '__main__':
    main()
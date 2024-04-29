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
import joblib
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from utils.dataset import VerseDataset

from metrics import dice_coef, batch_iou, mean_iou, iou_score
import losses
from utils.utils import str2bool, count_params
import pandas as pd
# 模型导入
import model.ResUnet.model as ResUnetModel
import model.unet.unet_model as UnetModel
import model.deeplab.deeplab_v3p as DeepLabModel
import model.DilatedUnet.model as DilatedUnetModel
import model.DenseUnet.model as DenseUnetModel
import model.DAUnet.DAU_model as DAUnetModel
import model.AUnet.model as AttentionUnetModel

arch_names = list(DAUnetModel.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')
# test = list(DilatedUnetModel.__dict__.keys())

def parse_args():
    parser = argparse.ArgumentParser()
    # 模型
    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='DilatedResUnet',
                        choices=arch_names,
                        help='model architecture: ' +
                             ' | '.join(arch_names) +
                             ' (default: NestedUNet)')
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    # 数据集
    parser.add_argument('--dataset', default="VerseData",
                        help='dataset name')
    # 输入通道数(B,C,H,W)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    # 文件类型
    parser.add_argument('--image-ext', default='png',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='png',
                        help='mask file extension')
    # 数据增强
    parser.add_argument('--aug', default=False, type=str2bool)
    # 损失函数
    parser.add_argument('--loss', default='LovaszHingeLoss',
                        choices=loss_names,
                        help='loss: ' +
                             ' | '.join(loss_names) +
                             ' (default: BCEDiceLoss)')
    # epochs次数
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    # 早过拟合停止
    parser.add_argument('--early-stop', default=60, type=int,
                        metavar='N', help='early stopping (default: 20)')
    # Batch_Size
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    # 优化器
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    # 学习率
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    # 增量
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    # 权重衰减
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    # 牛顿动量法
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()
    return args


class AverageMeter(object):
    """
    计算保存参数变量的更新
    初始化的时候就调用的reset()
    调用该类对象的update()的时候就会进行变量更新
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    ious = AverageMeter()
    model.train()
    # 遍历数组对象组合为一个序列索引
    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        target = target.cuda()
        # 计算输出
        if args.deepsupervision:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)

        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))

        # 计算梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])
    return log


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    ious = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            if args.deepsupervision:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)
            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log


def main():
    args = parse_args()
    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_%s_withDS' % (args.dataset, args.arch, args.loss)
        else:
            args.name = '%s_%s_%s_withoutDS' % (args.dataset, args.arch, args.loss)
    if not os.path.exists('trained_models/%s' % args.name):
        os.makedirs('trained_models/%s' % args.name)
    # 记录参数到文件
    print('Config --------')
    for arg in vars(args):
        print('%s,%s' % (arg, getattr(args, arg)))
    print('---------------')

    with open("trained_models/%s/args.txt" % args.name, 'w') as f:
        for arg in vars(args):
            print('%s,%s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'trained_models/%s/args.pkl' % args.name)

    # 定义损失函数
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[args.loss]().cuda()
    # 提升效率
    cudnn.benchmark = True

    # 数据集载入
    img_paths = glob(r'F:\Verse_Data\train_data_256x256\img\*')
    mask_paths = glob(r'F:\Verse_Data\train_data_256x256\mask\*')
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)
    print("train_nums:%s" % str(len(train_img_paths)))
    print("val_nums:%s" % str(len(val_img_paths)))

    # 创建模型
    print("=> creating model: %s " % args.arch)
    # 修改此处，即为修改模型
    trainModel = DAUnetModel.__dict__[args.arch]()
    trainModel = trainModel.cuda()
    params_model = count_params(trainModel) / (1024 * 1024)
    print("参数：%.2f" % (params_model) + "MB")
    with open("trained_models/%s/args.txt" % args.name, 'a') as f:
        print('params-count:%s' % (params_model) + "MB", file=f)

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, trainModel.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, trainModel.parameters()), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    train_dataset = VerseDataset(args, train_img_paths, train_mask_paths, args.aug)
    val_dataset = VerseDataset(args, val_img_paths, val_mask_paths, args.aug)

    # drop_last扔掉最后一个batch_size剩下的data
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'
    ])
    best_iou = 0
    trigger = 0
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch, args.epochs))

        # train
        train_log = train(args, train_loader, trainModel, criterion, optimizer, epoch)
        # val
        val_log = validate(args, val_loader, trainModel, criterion)

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['iou'],
            val_log['loss'],
            val_log['iou'],
        ], index=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('trained_models/%s/log.csv' % args.name, index=False)
        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(trainModel.state_dict(), 'trained_models/%s/model.pth' % args.name)
            best_iou = val_log['iou']
            print('=> saved best model')
            # 并保持当前最好的保存checkpoint
            checkpoint = {"model_state_dict": trainModel.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch}
            path_checkpoint = "trained_models/%s/checkpoint_%d_epoch.pkl" % (args.name, epoch)
            torch.save(checkpoint, path_checkpoint)
            trigger = 0
        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
    # print(arch_names)
    # print(test)
    # print(losses)

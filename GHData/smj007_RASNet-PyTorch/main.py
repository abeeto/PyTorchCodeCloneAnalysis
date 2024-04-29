'''
Paper         : RASNet: Segmentation for Tracking Surgical Instruments in Surgical Videos Using Refined Attention Segmentation Network
Authors       : Zhen-Liang Ni, Gui-Bin Bian, Xiao-Liang Xie, Zeng-Guang Hou, Xiao-Hu Zhou, Yan-Jie Zhou
Code Author    : Sai Mitheran Jagadesh Kumar

THIS IS AN UNOFFICIAL IMPLEMENTATION, VALUES VARY FROM THOSE REPORTED IN THE PAPER
Experiments carried out on Endovis18, using PixAcc instead of Dice
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
from glob import glob

from torch import autograd, optim
from tqdm import tqdm
import datetime
import math
from tensorboardX import SummaryWriter

from utils import *
from dataset import MICCAI_Dataset
from loss import CEL_Jaccard
from model import RASNet

def val_fn(model, criterion, valid_loader, num_classes):
    def eval_batch(model, image, target):
        image = image.to(device)
        pred = model(image)
        target = target.to(device)
        loss = criterion(pred, target)
        correct, labeled = batch_pix_accuracy(pred.data, target)
        inter, union = batch_intersection_union(pred, target, num_classes)
        return correct, labeled, inter, union, loss

    model.eval()
    test_loss = 0.0
    total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
    tbar = tqdm(valid_loader, desc='\r')
    for i, (image, target) in enumerate(tbar):
        with torch.no_grad():
            correct, labeled, inter, union, t_loss = eval_batch(model, image, target)

        total_correct += correct
        total_label += labeled
        total_inter += inter
        total_union += union
        test_loss += t_loss
        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        mIoU = IoU.mean()
        tbar.set_description(
            'pixAcc: %.3f, mIoU: %.3f, Val-loss: %.3f' % (pixAcc, mIoU, test_loss/(i + 1)))


def train_model(model, criterion, optimizer, train_loader, val_loader, num_classes, num_epochs=150):
    loss_list=[]
    logs_dir = './logs/T{}/'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(logs_dir)
    writer = SummaryWriter(logs_dir)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        dt_size = len(train_loader.dataset)
        tq = tqdm(total=math.ceil(dt_size/batch_size))
        tq.set_description('Epoch {}'.format(epoch))
        epoch_loss =[]
        step = 0

        # Training Start
        for x, y in train_loader:
            step += 1
            inputs = x.cuda()
            labels = y.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
  
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            tq.update(1)
            epoch_loss.append(loss.item())
            epoch_loss_mean = np.mean(epoch_loss).astype(np.float64)
            tq.set_postfix(loss='{0:.3f}'.format(epoch_loss_mean))

        loss_list.append(epoch_loss_mean)
        tq.close()
        print("Epoch %d Loss:%0.3f" % (epoch, epoch_loss_mean))

        # Validation Start
        val_fn(model, criterion, val_loader, num_classes)
        writer.add_scalar('Loss', epoch_loss_mean, epoch)

        # Adaptive LR 
        adjust_learning_rate(optimizer, epoch)
        # Save model weights
        torch.save(model.state_dict(), logs_dir + 'weight_{}.pth'.format(epoch)) # use model.module.state_dict() if Parallelized

        # Logging in txt files
        fileObject = open(logs_dir + 'loss_list.txt', 'w')
        
        for ip in loss_list:
            fileObject.write(str(ip))
            fileObject.write('\n')
        fileObject.close()

    writer.close()
    return model     

def run_model(train_loader, val_loader):
    model = RASNet().cuda()
    criterion = CEL_Jaccard()
    optimizer = optim.Adam(model.parameters(), lr=lr_base)

    train_model(model, criterion, optimizer, train_loader, val_loader, num_classes)

if __name__ == "__main__":
    res = models.resnet50(pretrained=False)
    num_classes = 8
    lr_base = 0.0003
    batch_size = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # It is recommended to download the full Endovis17/18 dataset to train and validate the model
    train_dataset = MICCAI_Dataset(
                data_root='gr_mtl_ssu_dataset/dataset/', 
                seq_set=[1, 5], 
                is_train=True)

    val_dataset = MICCAI_Dataset(
                data_root='gr_mtl_ssu_dataset/dataset/', 
                seq_set=[16], 
                is_train=False)

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True, num_workers=2,
                            drop_last=True)
                                
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False, num_workers=2,
                            drop_last=True)           


    run_model(train_loader, val_loader)      
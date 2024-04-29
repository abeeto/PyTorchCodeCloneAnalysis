import os
import sys
import cv2
import shutil
import warnings
from model import SANet
from utils import SSIM_Loss
from data import MyDataset

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


def main():
    model = SANet().cuda() if torch.cuda.is_available() else SANet()
    train_img_file_path = os.path.join('StoneData', 'train', 'images')
    train_gt_file_path = os.path.join('StoneData', 'train', 'ground_truth')
    val_img_file_path = os.path.join('StoneData', 'val', 'images')
    val_gt_file_path = os.path.join('StoneData', 'val', 'ground_truth')
    train_loader = DataLoader(MyDataset(train_img_file_path, train_gt_file_path), batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(MyDataset(val_img_file_path, val_gt_file_path), batch_size=1, num_workers=4)
    # criterion = nn.MSELoss(size_average=False).cuda() if torch.cuda.is_available() else nn.MSELoss(size_average=False)
    criterion = SSIM_Loss().cuda() if torch.cuda.is_available() else SSIM_Loss()
    # criterion = SANetLoss(1).cuda() if torch.cuda.is_available() else SANetLoss(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for i in range(10):
        train(model, train_loader, criterion, optimizer, i + 1)
        validate(model, val_loader, criterion)

def train(model, train_loader, criterion, optimizer, epoch):
    cuda_stat = torch.cuda.is_available()
    step_num = 0
    total_loss = 0.0
    step_loss = 0.0
    model.train()
    for i, (img, gt) in enumerate(train_loader):
        img = Variable(img.cuda()) if cuda_stat else Variable(img)
        gt = Variable(gt.cuda()) if cuda_stat else Variable(gt)
        optimizer.zero_grad()
        out_put = model(img)
        loss = criterion(out_put, gt)
        step_loss = loss.item()
        total_loss += step_loss
        loss.backward()
        optimizer.step()
        step_num = step_num + 1
        print("Epoch %d: step %d ends. loss is %.4f" % (epoch, step_num, step_loss))
    print("Epoch %d ended: average loss is %.4f" % (epoch, total_loss / step_num))



def validate(model, val_loader, criterion):
    cuda_stat = torch.cuda.is_available()
    total_loss = 0.0
    total_step = 0
    model.eval()
    for i, (img, gt) in enumerate(val_loader):
        img = Variable(img.cuda()) if cuda_stat else Variable(img)
        gt = Variable(gt.cuda()) if cuda_stat else Variable(gt)
        with torch.no_grad():
            out_put = model(img)
        loss = criterion(out_put, gt)
        total_loss += loss.item()
        print(loss.item())
        total_step += 1
    print("Validation ended: Average loss is %.4f" % (total_loss / total_step))



if __name__ == '__main__':
    main()
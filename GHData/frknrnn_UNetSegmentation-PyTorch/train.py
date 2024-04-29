import argparse
import os
import copy

import cv2
import numpy as np
import torch
import glob, itertools
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from Dataset import ImageDataset
from Models.UNet.unet_model import UNet
from utils import AverageMeter
from PIL import Image
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    train_path_raw = "C:/Users/PC/Documents/PyCharm Project/CellAnalyzer/DataSets/UnetData/orginal_crop"
    train_path_label = "C:/Users/PC/Documents/PyCharm Project/CellAnalyzer/DataSets/UnetData/label_crop"
    output_dir = "C:/Users/PC/PycharmProjects/Pytorch-SegmentationModels/output"
    batch_size = 2
    num_workers = 8
    num_epochs = 400
    lr_ = 1e-4


    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    torch.manual_seed(123)


    model = UNet(1, 1).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_)

    train_paths_raw = sorted(glob.glob(train_path_raw + "/*.*"))
    train_paths_label = sorted(glob.glob(train_path_label + "/*.*"))

    test_paths_raw  = train_paths_raw[0:5000]
    test_paths_label  = train_paths_label[0:5000]

    train_dataset = ImageDataset(files_raw=train_paths_raw,files_label=train_paths_label)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  pin_memory=True,
                                  drop_last=True)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_losses = AverageMeter()
        with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, num_epochs - 1))
            for data in train_dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
        torch.save(model.state_dict(), os.path.join(output_dir, 'epoch_{}.pth'.format(epoch)))







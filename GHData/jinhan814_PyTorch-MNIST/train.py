import argparse
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader.dataset import MnistDataset
from model.model import MnistModule
from torch.utils.data import DataLoader
from torchvision import transforms


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def save_model(model, save_dir, file_name):
    path = os.path.join(save_dir, file_name)
    torch.save(model, path)


def train(epochs, lr, batch_size):
    set_seed(814)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = MnistDataset('/opt/ml/MNIST/data/train')
    val_dataset   = MnistDataset('/opt/ml/MNIST/data/val')
    train_loader  = DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               drop_last=True)
    val_loader    = DataLoader(dataset=val_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               drop_last=True)

    model = MnistModule().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                     T_max=10,
                                                     eta_min=0.001)

    for epoch in range(epochs):
        model.train()
        avg_loss = 0
        for x, label in train_loader:
            x      = x.unsqueeze(axis=1).to(device)
            label  = label.to(device)
            output = model(x)
            loss   = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss
        scheduler.step()
        avg_loss /= len(train_loader)
        cur_lr = optimizer.param_groups[0]['lr']
        print(f'[Epoch: {epoch+1:>4}] train_loss: {avg_loss:>.6} lr: {cur_lr:>.6}')

        with torch.no_grad():
            model.eval()
            avg_loss, acc = 0, 0
            for x, label in val_loader:
                x      = x.unsqueeze(axis=1).to(device)
                label  = label.to(device)
                output = model(x)
                loss   = criterion(output, label)
                avg_loss += loss
                acc += int(torch.sum(torch.argmax(output, dim=1) == label))
            avg_loss /= len(val_loader)
            acc       = acc / len(val_dataset) * 100
            print(f'[Epoch: {epoch+1:>4}] val_loss  : {avg_loss:>.6} acc: {acc:>.6}')
        
        save_model(model, '/opt/ml/MNIST/model_save', f'epoch_{epoch+1}.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch-size', '-b', type=int, default=50)
    args = parser.parse_args()
    train(args.epoch, args.lr, args.batch_size)

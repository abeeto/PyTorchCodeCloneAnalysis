import torch      
import torch.nn as nn
from torchvision import datasets, models, transforms     
import torchvision.transforms as transforms              
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
import cv2
import time
import sys
import os
import copy
import GPUtil
import shutil
import csv
from config import configDict

from DefectDataset_CAM_ICT import DefectDataset, DataSampler
from network_cam_ict import *
from visdom import Visdom
from matplotlib import pyplot as plt
import utils
from PIL import Image
from matplotlib import cm
from numpy.random import default_rng

from losses import kl_loss
from metrics import *


def create_model(ema = False):
    model = network1(n_class=num_classes)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model



# CONFIG VARIABLES
# Dataset parameters
root_dir = configDict['root_dir']
num_classes = configDict['num_classes']
# Training and optimization parameters
epochs = configDict['epochs']
lr = configDict['lr']
momentum = configDict['momentum']
optim_w_decay = configDict['optim_w_decay']
step_size = configDict['step_size']
gamma = configDict['gamma']
# Admin
load_model = configDict['load_ckp']

# Initialize plotter
global plotter
plotter = utils.VisdomLinePlotter(env_name='main')

# Create results directories
savedir = configDict['directory_name']
print("Savedir:", savedir)
model_dir = os.path.join('models/', savedir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
checkpoint_dir = os.path.join(model_dir, 'checkpoints/')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
best_dir = os.path.join(model_dir, 'best/')
if not os.path.exists(best_dir):
    os.makedirs(best_dir)

# create config file
csv_file = model_dir + '/config.txt'
f = open(csv_file, 'w')
f.write(str(configDict))
f.close()

# Activate GPU
use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Dataloaders
dataset = {
    'train': DefectDataset(root_dir = root_dir, image_set = 'labeled', n_class = num_classes),
    'val': DefectDataset(root_dir = root_dir, image_set='val', n_class=num_classes)
}
dataloader = {
    'train': DataLoader(dataset['train'], batch_size = 10),
    'val': DataLoader(dataset['val'], batch_size= 10)
}

# Network
encoder = VGGNet(pretrained=True, n_class = num_classes)
encoder = encoder.to(device)
student = create_model()
student = student.to(device)

# Optimizers
optimizer = torch.optim.SGD([
    {'params': encoder.parameters()},
    {'params': student.parameters()}
    ],
                            lr=lr,
                            momentum=momentum,
                            weight_decay=optim_w_decay,
                            nesterov=False)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of gamma every step_size epochs

if load_model:
    print("Loading checkpoint from previous save file...")
    ckp_path = checkpoint_dir + 'checkpoint.pt'
    encoder, student, optimizer, epoch = utils.load_ckp(ckp_path, encoder, student,teacher=None, optimizer=optimizer)
    print("Epoch loaded: ", epoch)
# supervised loss
sup_loss = nn.BCEWithLogitsLoss()
global global_step
global_step = 0
best_IU = 0
for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch,epochs - 1))
    for phase in ['train','val']:
        batchIU = []
        batchF1 = []
        batchF1_ = []
        running_acc = 0
        running_loss = 0
        print("Current phase: ", phase)
        if phase == 'train':
            encoder.train()
            student.train()
        else:
            encoder.eval()
            student.eval()
        # Train/val loop
        for iter, (input, target) in enumerate(dataloader[phase]):
            input = input.to(device)
            target = target.to(device)

            if phase == 'train':
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    enc_out = encoder(input, cams = False) # pass all inputs through encoder first
                    sup_out = student(enc_out)
                loss = sup_loss(sup_out, target)
                sup_out_ = sup_out.detach().clone()
                target_ = target.detach().clone()
                sup_out__ = sup_out_.argmax(dim = 1).cpu()
                target__ = target_.argmax(dim = 1).cpu()
                accuracy = utils.pixel_accuracy(sup_out_, target_)
                iou = utils.iou(sup_out_, target_)
                batchIU.append(iou)
                batchF1.append(utils.f1(iou))
                running_acc += np.mean(accuracy)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                global_step += 1

                # print("Loss: ", loss.item(), "Batch_IU: ", iou, "Batch_F1:", utils.f1(iou))        

            else:
                with torch.no_grad():
                    enc_out = encoder(input, cams = False)
                    out = student(enc_out)
                    # Metrics:
                    accuracy = utils.pixel_accuracy(out,target) # batch accuracy
                    iou = utils.iou(out, target)
                    batchIU.append(iou) # batch IU
                    batchF1.append(utils.f1(iou))
                    loss = sup_loss(out, target)
                    running_acc += np.mean(accuracy)
                    running_loss +=  loss.item()
        epoch_acc = running_acc/(iter+1)
        epoch_loss = running_loss/(iter + 1)
        epoch_IU = np.mean(batchIU)
        epoch_F1 = np.mean(batchF1)
        if phase == 'train':
            scheduler.step()
        print('{} Loss: {:.4f}, Acc: {:.4f}, IoU: {:.4f}, F1: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_IU, epoch_F1))
        
        plotter.plot('loss', phase, 'Loss', epoch, epoch_loss)
        plotter.plot('acc', phase, 'Accuracy', epoch, epoch_acc)
        plotter.plot('IU', phase, 'IU', epoch, epoch_IU)
        plotter.plot('F1', phase, 'F1', epoch, epoch_F1)

        if phase == 'val' and epoch_IU > best_IU:
            best_IU = epoch_IU
            is_best = True
        else:
            is_best = False
        checkpoint = {
            'epoch': epoch + 1,
            'encoder_state_dict': encoder.state_dict(),
            'student_state_dict': student.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        utils.save_ckp(checkpoint, is_best, checkpoint_dir, best_dir)       




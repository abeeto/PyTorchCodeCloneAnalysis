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
# from config import configDict

from DefectDataset_CAM_ICT import DefectDataset, DataSampler
from network_cam_ict import *
from visdom import Visdom
from matplotlib import pyplot as plt
from utils import *
from PIL import Image
from matplotlib import cm
from numpy.random import default_rng

from losses import kl_loss
import time 

def create_model(ema = False):
    model = network1(n_class = num_classes)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model




imgs = []
gts = []
p5s = []
p10s = []
p20s = []
p50s = []
pfulls = []


# 5Percent Labeled
configDict = {
    'directory_name': 'welding_5Percent_Labeled',
    'root_dir' : 'data/Welding/',
    'num_classes' : 2,
    'labeled_batch_size' : 2,
    'unlabeled_batch_size': 2,
    'epochs' : 300,
    'lr' : 1e-2,
    'momentum' : 0.9,
    'optim_w_decay' : 1e-5,
    'step_size' : 200,
    'gamma' : 0.1,
    'ema_decay': 0.99,
    'alpha_ict': 0.5,
    'num_labeled': 65, 
    'num_unlabeled': 70,
    'consistency_weight': 'none', # Options: 'none', 'ramp', 'dynamic' 
    'load_ckp': True,
    'print_gpu_usage': False
}

# CONFIG VARIABLES
# Dataset parameters
root_dir = configDict['root_dir']
num_classes = configDict['num_classes']
labeled_batch_size = configDict['labeled_batch_size']
unlabeled_batch_size = configDict['unlabeled_batch_size']
# Training and optimization parameters
epochs = configDict['epochs']
lr = configDict['lr']
momentum = configDict['momentum']
optim_w_decay = configDict['optim_w_decay']
step_size = configDict['step_size']
gamma = configDict['gamma']
consistency_weight = configDict['consistency_weight']
# Mean Teacher parameters
ema_decay = configDict['ema_decay']
# ICT parameters
alpha_ict = configDict['alpha_ict']
# Admin
load_model = configDict['load_ckp']

assert load_model, "Cannot load model. (load_model = False)"

loaddir = configDict['directory_name']
print("Current Directory:", loaddir)
model_dir = os.path.join('models/', loaddir)
checkpoint_dir = os.path.join(model_dir, 'checkpoints/')
best_dir = os.path.join(model_dir, 'best/')
# save_dir = 'results/' + loaddir + '/' 
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)


# Activate GPU
use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Dataloading
val_dataset = DefectDataset(root_dir = root_dir, image_set='val', n_class=num_classes, mode = 'inference')
val_dataloader = DataLoader(val_dataset, batch_size= 1)

encoder = VGGNet(pretrained=True, n_class = num_classes)
student = create_model()
best_path = best_dir + 'best_model.pt'
encoder, student, epoch = load_ckp(best_path, encoder, student)
print("Epoch loaded: ", epoch)
encoder = encoder.to(device)
student = student.to(device)
encoder.eval()
student.eval()
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
print("Validating at epoch: {:.4f}".format(epoch))
with torch.no_grad():
    for iter, (input, target, filename) in enumerate(val_dataloader):
        filename = [os.path.basename(f) for f in filename]
        input = input.to(device)
        target = target.to(device)
        enc_out = encoder(input, cams = False)
        out = student(enc_out)
        N, _, h, w = out.shape
        out = out.detach().cpu().numpy()
        pred = out.transpose(0, 2, 3, 1).reshape(-1, configDict['num_classes']).argmax(axis=1).reshape(N, h, w)
        gt = target.cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, configDict['num_classes']).argmax(axis=1).reshape(N, h, w)
        img = input.detach().cpu()
        img = img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
        img = img.numpy().transpose(0,2,3,1)
        pred = normalize(pred)
        gt = normalize(gt)
        # save_predictions(imgList = [img[0], gt[0], pred[0]], path = save_dir + filename[0])
        imgs.append(img)
        gts.append(gt)
        p5s.append(pred)
############################################################
# 15 Percent Labeled
# CONFIG VARIABLES
configDict = {
    'directory_name': 'welding_15Percent_Labeled',
    'root_dir' : 'data/Welding/',
    'num_classes' : 2,
    'labeled_batch_size' : 2,
    'unlabeled_batch_size': 2,
    'epochs' : 300,
    'lr' : 1e-2,
    'momentum' : 0.9,
    'optim_w_decay' : 1e-5,
    'step_size' : 200,
    'gamma' : 0.1,
    'ema_decay': 0.99,
    'alpha_ict': 0.5,
    'num_labeled': 65, 
    'num_unlabeled': 70,
    'consistency_weight': 'none', # Options: 'none', 'ramp', 'dynamic' 
    'load_ckp': True,
    'print_gpu_usage': False
}

# Dataset parameters
root_dir = configDict['root_dir']
num_classes = configDict['num_classes']
labeled_batch_size = configDict['labeled_batch_size']
unlabeled_batch_size = configDict['unlabeled_batch_size']
# Training and optimization parameters
epochs = configDict['epochs']
lr = configDict['lr']
momentum = configDict['momentum']
optim_w_decay = configDict['optim_w_decay']
step_size = configDict['step_size']
gamma = configDict['gamma']
consistency_weight = configDict['consistency_weight']
# Mean Teacher parameters
ema_decay = configDict['ema_decay']
# ICT parameters
alpha_ict = configDict['alpha_ict']
# Admin
load_model = configDict['load_ckp']

assert load_model, "Cannot load model. (load_model = False)"

loaddir = configDict['directory_name']
print("Current Directory:", loaddir)
model_dir = os.path.join('models/', loaddir)
checkpoint_dir = os.path.join(model_dir, 'checkpoints/')
best_dir = os.path.join(model_dir, 'best/')
# save_dir = 'results/' + loaddir + '/' 
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)


# Activate GPU
use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Dataloading
val_dataset = DefectDataset(root_dir = root_dir, image_set='val', n_class=num_classes, mode = 'inference')
val_dataloader = DataLoader(val_dataset, batch_size= 1)

encoder = VGGNet(pretrained=True, n_class = num_classes)
student = create_model()
best_path = best_dir + 'best_model.pt'
encoder, student, epoch = load_ckp(best_path, encoder, student)
print("Epoch loaded: ", epoch)
encoder = encoder.to(device)
student = student.to(device)
encoder.eval()
student.eval()
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
print("Validating at epoch: {:.4f}".format(epoch))
with torch.no_grad():
    for iter, (input, target, filename) in enumerate(val_dataloader):
        filename = [os.path.basename(f) for f in filename]
        input = input.to(device)
        target = target.to(device)
        enc_out = encoder(input, cams = False)
        out = student(enc_out)
        N, _, h, w = out.shape
        out = out.detach().cpu().numpy()
        pred = out.transpose(0, 2, 3, 1).reshape(-1, configDict['num_classes']).argmax(axis=1).reshape(N, h, w)
        gt = target.cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, configDict['num_classes']).argmax(axis=1).reshape(N, h, w)
        img = input.detach().cpu()
        img = img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
        img = img.numpy().transpose(0,2,3,1)
        pred = normalize(pred)
        gt = normalize(gt)
        # save_predictions(imgList = [img[0], gt[0], pred[0]], path = save_dir + filename[0])
        p10s.append(pred)


######################################
# 30 Percent_Labeled
# CONFIG VARIABLES
configDict = {
    'directory_name': 'welding_30Percent_Labeled',
    'root_dir' : 'data/Welding/',
    'num_classes' : 2,
    'labeled_batch_size' : 2,
    'unlabeled_batch_size': 2,
    'epochs' : 300,
    'lr' : 1e-2,
    'momentum' : 0.9,
    'optim_w_decay' : 1e-5,
    'step_size' : 200,
    'gamma' : 0.1,
    'ema_decay': 0.99,
    'alpha_ict': 0.5,
    'num_labeled': 65, 
    'num_unlabeled': 70,
    'consistency_weight': 'none', # Options: 'none', 'ramp', 'dynamic' 
    'load_ckp': True,
    'print_gpu_usage': False
}

# Dataset parameters
root_dir = configDict['root_dir']
num_classes = configDict['num_classes']
labeled_batch_size = configDict['labeled_batch_size']
unlabeled_batch_size = configDict['unlabeled_batch_size']
# Training and optimization parameters
epochs = configDict['epochs']
lr = configDict['lr']
momentum = configDict['momentum']
optim_w_decay = configDict['optim_w_decay']
step_size = configDict['step_size']
gamma = configDict['gamma']
consistency_weight = configDict['consistency_weight']
# Mean Teacher parameters
ema_decay = configDict['ema_decay']
# ICT parameters
alpha_ict = configDict['alpha_ict']
# Admin
load_model = configDict['load_ckp']

assert load_model, "Cannot load model. (load_model = False)"

loaddir = configDict['directory_name']
print("Current Directory:", loaddir)
model_dir = os.path.join('models/', loaddir)
checkpoint_dir = os.path.join(model_dir, 'checkpoints/')
best_dir = os.path.join(model_dir, 'best/')
# save_dir = 'results/' + loaddir + '/' 
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)


# Activate GPU
use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Dataloading
val_dataset = DefectDataset(root_dir = root_dir, image_set='val', n_class=num_classes, mode = 'inference')
val_dataloader = DataLoader(val_dataset, batch_size= 1)

encoder = VGGNet(pretrained=True, n_class = num_classes)
student = create_model()
best_path = best_dir + 'best_model.pt'
encoder, student, epoch = load_ckp(best_path, encoder, student)
print("Epoch loaded: ", epoch)
encoder = encoder.to(device)
student = student.to(device)
encoder.eval()
student.eval()
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
print("Validating at epoch: {:.4f}".format(epoch))
with torch.no_grad():
    for iter, (input, target, filename) in enumerate(val_dataloader):
        filename = [os.path.basename(f) for f in filename]
        input = input.to(device)
        target = target.to(device)
        enc_out = encoder(input, cams = False)
        out = student(enc_out)
        N, _, h, w = out.shape
        out = out.detach().cpu().numpy()
        pred = out.transpose(0, 2, 3, 1).reshape(-1, configDict['num_classes']).argmax(axis=1).reshape(N, h, w)
        gt = target.cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, configDict['num_classes']).argmax(axis=1).reshape(N, h, w)
        img = input.detach().cpu()
        img = img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
        img = img.numpy().transpose(0,2,3,1)
        pred = normalize(pred)
        gt = normalize(gt)
        # save_predictions(imgList = [img[0], gt[0], pred[0]], path = save_dir + filename[0])
        p20s.append(pred)

####################################
# 50 Percent Labeled
# CONFIG VARIABLES
configDict = {
    'directory_name': 'welding_50Percent_Labeled',
    'root_dir' : 'data/Welding/',
    'num_classes' : 2,
    'labeled_batch_size' : 2,
    'unlabeled_batch_size': 2,
    'epochs' : 300,
    'lr' : 1e-2,
    'momentum' : 0.9,
    'optim_w_decay' : 1e-5,
    'step_size' : 200,
    'gamma' : 0.1,
    'ema_decay': 0.99,
    'alpha_ict': 0.5,
    'num_labeled': 65, 
    'num_unlabeled': 70,
    'consistency_weight': 'none', # Options: 'none', 'ramp', 'dynamic' 
    'load_ckp': True,
    'print_gpu_usage': False
}

# Dataset parameters
root_dir = configDict['root_dir']
num_classes = configDict['num_classes']
labeled_batch_size = configDict['labeled_batch_size']
unlabeled_batch_size = configDict['unlabeled_batch_size']
# Training and optimization parameters
epochs = configDict['epochs']
lr = configDict['lr']
momentum = configDict['momentum']
optim_w_decay = configDict['optim_w_decay']
step_size = configDict['step_size']
gamma = configDict['gamma']
consistency_weight = configDict['consistency_weight']
# Mean Teacher parameters
ema_decay = configDict['ema_decay']
# ICT parameters
alpha_ict = configDict['alpha_ict']
# Admin
load_model = configDict['load_ckp']

assert load_model, "Cannot load model. (load_model = False)"

loaddir = configDict['directory_name']
print("Current Directory:", loaddir)
model_dir = os.path.join('models/', loaddir)
checkpoint_dir = os.path.join(model_dir, 'checkpoints/')
best_dir = os.path.join(model_dir, 'best/')
# save_dir = 'results/' + loaddir + '/' 
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)


# Activate GPU
use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Dataloading
val_dataset = DefectDataset(root_dir = root_dir, image_set='val', n_class=num_classes, mode = 'inference')
val_dataloader = DataLoader(val_dataset, batch_size= 1)

encoder = VGGNet(pretrained=True, n_class = num_classes)
student = create_model()
best_path = best_dir + 'best_model.pt'
encoder, student, epoch = load_ckp(best_path, encoder, student)
print("Epoch loaded: ", epoch)
encoder = encoder.to(device)
student = student.to(device)
encoder.eval()
student.eval()
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
print("Validating at epoch: {:.4f}".format(epoch))
with torch.no_grad():
    for iter, (input, target, filename) in enumerate(val_dataloader):
        filename = [os.path.basename(f) for f in filename]
        input = input.to(device)
        target = target.to(device)
        enc_out = encoder(input, cams = False)
        out = student(enc_out)
        N, _, h, w = out.shape
        out = out.detach().cpu().numpy()
        pred = out.transpose(0, 2, 3, 1).reshape(-1, configDict['num_classes']).argmax(axis=1).reshape(N, h, w)
        gt = target.cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, configDict['num_classes']).argmax(axis=1).reshape(N, h, w)
        img = input.detach().cpu()
        img = img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
        img = img.numpy().transpose(0,2,3,1)
        pred = normalize(pred)
        gt = normalize(gt)
        # save_predictions(imgList = [img[0], gt[0], pred[0]], path = save_dir + filename[0])
        p50s.append(pred)

###########################################
# Full
# CONFIG VARIABLES
configDict = {
    'directory_name': 'welding_full',
    'root_dir' : 'data/Welding/',
    'num_classes' : 2,
    'labeled_batch_size' : 2,
    'unlabeled_batch_size': 2,
    'epochs' : 300,
    'lr' : 1e-2,
    'momentum' : 0.9,
    'optim_w_decay' : 1e-5,
    'step_size' : 200,
    'gamma' : 0.1,
    'ema_decay': 0.99,
    'alpha_ict': 0.5,
    'num_labeled': 65, 
    'num_unlabeled': 70,
    'consistency_weight': 'none', # Options: 'none', 'ramp', 'dynamic' 
    'load_ckp': True,
    'print_gpu_usage': False
}

# Dataset parameters
root_dir = configDict['root_dir']
num_classes = configDict['num_classes']
labeled_batch_size = configDict['labeled_batch_size']
unlabeled_batch_size = configDict['unlabeled_batch_size']
# Training and optimization parameters
epochs = configDict['epochs']
lr = configDict['lr']
momentum = configDict['momentum']
optim_w_decay = configDict['optim_w_decay']
step_size = configDict['step_size']
gamma = configDict['gamma']
consistency_weight = configDict['consistency_weight']
# Mean Teacher parameters
ema_decay = configDict['ema_decay']
# ICT parameters
alpha_ict = configDict['alpha_ict']
# Admin
load_model = configDict['load_ckp']

assert load_model, "Cannot load model. (load_model = False)"

loaddir = configDict['directory_name']
print("Current Directory:", loaddir)
model_dir = os.path.join('models/', loaddir)
checkpoint_dir = os.path.join(model_dir, 'checkpoints/')
best_dir = os.path.join(model_dir, 'best/')
save_dir = 'results/' + 'Welding/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# Activate GPU
use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Dataloading
val_dataset = DefectDataset(root_dir = root_dir, image_set='val', n_class=num_classes, mode = 'inference')
val_dataloader = DataLoader(val_dataset, batch_size= 1)

encoder = VGGNet(pretrained=True, n_class = num_classes)
student = create_model()
best_path = best_dir + 'best_model.pt'
encoder, student, epoch = load_ckp(best_path, encoder, student)
print("Epoch loaded: ", epoch)
encoder = encoder.to(device)
student = student.to(device)
encoder.eval()
student.eval()
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
print("Validating at epoch: {:.4f}".format(epoch))
with torch.no_grad():
    for iter, (input, target, filename) in enumerate(val_dataloader):
        filename = [os.path.basename(f) for f in filename]
        input = input.to(device)
        target = target.to(device)
        enc_out = encoder(input, cams = False)
        out = student(enc_out)
        N, _, h, w = out.shape
        out = out.detach().cpu().numpy()
        pred = out.transpose(0, 2, 3, 1).reshape(-1, configDict['num_classes']).argmax(axis=1).reshape(N, h, w)
        gt = target.cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, configDict['num_classes']).argmax(axis=1).reshape(N, h, w)
        img = input.detach().cpu()
        img = img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
        img = img.numpy().transpose(0,2,3,1)
        pred = normalize(pred)
        gt = normalize(gt)
        pfulls.append(pred)

count = 0
for img, gt, p5, p10, p20, p50, pfull in zip(imgs, gts, p5s, p10s, p20s, p50s, pfulls):
    save_predictions(imgList = [img[0], gt[0], p5[0], p10[0], p20[0], p50[0], pfull[0]], path = save_dir + str(count) + '.png')
    count += 1


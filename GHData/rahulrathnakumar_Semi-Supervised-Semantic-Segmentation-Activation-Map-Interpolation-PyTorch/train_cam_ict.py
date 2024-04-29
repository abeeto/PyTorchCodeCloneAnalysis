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
from getPCA import *



def create_model(ema = False):
    model = network1(n_class=num_classes)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def interpolation(u1, u2, lam):
    u = lam*u1 + (1-lam)*u2
    return u


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
if not load_model:
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
    'train': DefectDataset(root_dir = root_dir, image_set = 'train', n_class = num_classes),
    'val': DefectDataset(root_dir = root_dir, image_set='val', n_class=num_classes)
}
sampler = DataSampler(dataset = dataset['train'], root_dir = root_dir, image_set = 'train',
                        labeled_batch_size = labeled_batch_size, unlabeled_batch_size = unlabeled_batch_size)

dataloader = {
    'train': DataLoader(dataset['train'], batch_sampler = sampler),
    'val': DataLoader(dataset['val'], batch_size= 10)
}

# Network
encoder = VGGNet(pretrained=True, n_class = num_classes)
encoder = encoder.to(device)
student = create_model()
teacher = create_model(ema = True)
student = student.to(device)
teacher = teacher.to(device)

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
    encoder, student, teacher, optimizer, epoch = utils.load_ckp(ckp_path, encoder, student, teacher, optimizer)

# supervised loss
sup_loss = nn.BCEWithLogitsLoss()

# random number generator
rng = default_rng()

global global_step
global_step = 0
best_IU = 0
for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch,epochs - 1))
    for phase in ['train','val']:
        batchIU = []
        batchF1 = []
        batchF1_tol = []
        running_acc = 0
        running_loss = 0
        running_consistency_loss = 0
        running_supervised_loss = 0
        print("Current phase: ", phase)
        if phase == 'train':
            encoder.train()
            student.train()
            teacher.train()
        else:
            encoder.eval()
            student.eval()
        # Train/val loop
        for iter, (input, target) in enumerate(dataloader[phase]):
            F1_tol = [] # This is per  batch
            input = input.to(device)
            target = target.to(device)

            if phase == 'train':
                optimizer.zero_grad()
                labeled_indices = utils.getLabeledInBatch(target)
                unlabeled_indices = np.setdiff1d(np.arange(input.size()[0]), labeled_indices)
                labeled_targets = target[labeled_indices]
                lam = rng.beta(alpha_ict, alpha_ict)
                with torch.set_grad_enabled(True):
                    cam, enc_out = encoder(input, cams = True) # pass all inputs through encoder first
                    labeled_enc_out = [e[labeled_indices] for e in enc_out]
                    unlabeled_enc_out = [e[unlabeled_indices] for e in enc_out]
                    labeled_cam = [c[labeled_indices] for c in cam]
                    unlabeled_cam = [c[unlabeled_indices] for c in cam]
                    assert unlabeled_batch_size % 2 == 0, "Unlabeled batch size is not exactly divisible by 2."
                    unlabeled_cam_1 = [c[0: unlabeled_batch_size//2] for c in unlabeled_cam]
                    unlabeled_cam_2 = [c[unlabeled_batch_size//2: unlabeled_batch_size] for c in unlabeled_cam]
                    unlabeled_enc_out_1 = [u[0:unlabeled_batch_size//2] for u in unlabeled_enc_out]
                    unlabeled_enc_out_2 = [u[unlabeled_batch_size//2:unlabeled_batch_size] for u in unlabeled_enc_out]
                    
                    sup_out = student(labeled_enc_out)
                    mix = [interpolation(c1, c2, lam) for c1, c2 in zip(unlabeled_cam_1, unlabeled_cam_2)]
                    mix_out = student(mix, cam_in = True)
                with torch.no_grad():
                    unlabeled_out_1 = teacher(unlabeled_enc_out_1)
                    unlabeled_out_2 = teacher(unlabeled_enc_out_2)
                prob_unlabeled_out_1 = F.softmax(unlabeled_out_1, dim = 1)
                prob_unlabeled_out_2 = F.softmax(unlabeled_out_2, dim = 1)
                prob_mix_out = F.log_softmax(mix_out, dim = 1)
                prob_unlabeled_out_mix = interpolation(prob_unlabeled_out_1, prob_unlabeled_out_2, lam)
                if consistency_weight == 'ramp':
                    consistency_weight = utils.get_current_consistency_weight(epoch)
                elif consistency_weight == 'none':
                    consistency_weight = 1
                l_cons = consistency_weight*kl_loss(input_probs=prob_mix_out, target_probs=prob_unlabeled_out_mix)
                l_sup = sup_loss(sup_out, labeled_targets)
                loss = l_sup + l_cons
                accuracy = utils.pixel_accuracy(sup_out, labeled_targets)
                iou = utils.iou(sup_out, labeled_targets)
                batchIU.append(iou)
                batchF1.append(utils.f1(iou))
                running_acc += np.mean(accuracy)
                running_loss += loss.item()
                running_consistency_loss += l_cons.item()
                running_supervised_loss += l_sup.item()
                loss.backward()
                optimizer.step()
                global_step += 1

                # print("Loss: ", loss.item(), " Loss_Supervised: ", l_sup.item(), 
                #     "Loss_Consistency: ", l_cons.item(), "Consistency weight: ", consistency_weight)        
            else:
                with torch.no_grad():
                    enc_out = encoder(input, cams = False)
                    out = student(enc_out)
                    # Metrics:
                    accuracy = utils.pixel_accuracy(out,target) # batch accuracy
                    iou = utils.iou(out, target)
                    batchIU.append(iou) # batchIU
                    batchF1.append(utils.f1(iou))
                    loss = sup_loss(out, target)
                    running_acc += np.mean(accuracy)
                    running_loss +=  loss.item()
        epoch_acc = running_acc/(iter+1)
        epoch_loss = running_loss/(iter + 1)
        epoch_IU = np.mean(batchIU)
        epoch_F1 = np.mean(batchF1)
        if phase == 'train':
            epoch_loss_consistency = running_consistency_loss/(iter + 1)
            epoch_loss_supervised = running_supervised_loss/(iter + 1)
            utils.update_ema_variables(student, teacher, ema_decay, global_step)
            scheduler.step()
        print('{} Loss: {:.4f}, Acc: {:.4f}, IoU: {:.4f}, F1: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_IU, epoch_F1))
        
        plotter.plot('loss', phase, 'Loss', epoch, epoch_loss)
        if phase == 'train':
            plotter.plot('loss_consistency', phase, 'Consistency Loss', epoch, epoch_loss_consistency)
            plotter.plot('loss_supervised', phase, 'Supervised Loss', epoch, epoch_loss_supervised)
        plotter.plot('acc', phase, 'Accuracy', epoch, epoch_acc)
        plotter.plot('IU', phase, 'IU', epoch, epoch_IU)
        plotter.plot('F1', phase, 'F1', epoch, epoch_F1)
        # plotter.plot('F1-Tolerance', phase, 'F1-Tolerance', epoch, epoch_F1_tol)

        if phase == 'val' and epoch_IU > best_IU:
            best_IU = epoch_IU
            is_best = True
        else:
            is_best = False
        checkpoint = {
            'epoch': epoch + 1,
            'encoder_state_dict': encoder.state_dict(),
            'student_state_dict': student.state_dict(),
            'teacher_state_dict': teacher.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        utils.save_ckp(checkpoint, is_best, checkpoint_dir, best_dir)       




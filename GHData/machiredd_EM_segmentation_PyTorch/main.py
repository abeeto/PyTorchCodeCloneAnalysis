# import torchvision.models.segmentation as models
from model_resunet import *
import torch.nn as nn
from dataset_in import EM_dataset, RandomGenerator
from torchvision import transforms
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from losses import DiceLoss
from simple_unet import UNet
import cv2
import logging
import sys
from utils import save_model, save_plots, SaveBestModel
from helper_fns_pytorch import train_model_new,evaluate_model

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")
dest_dir = '/home/groups/graylab_share/OMERO.rdsStore/machired/EM5/pytorch_trials/resunet/output/'
root_path = '/home/groups/graylab_share/OMERO.rdsStore/machired/EM5/pytorch_trials/resunet/data/datasets/101a_dataset/data/'
in_dir = 'images'
tar_dir = 'nuclei'
img_size = 512
seed = 1234

db_train = EM_dataset(base_dir=root_path, in_dir_name=in_dir, out_dir_name=tar_dir)
# db_train = EM_dataset(base_dir=root_path, in_dir_name=in_dir, out_dir_name=tar_dir,
#                                transform=transforms.Compose(
#                                    [RandomGenerator(output_size=[img_size, img_size])]))
print("The length of train set is: {}".format(len(db_train)))
trainloader = DataLoader(db_train, batch_size=6, shuffle=True, num_workers=0, pin_memory=True, drop_last = True)

db_test = EM_dataset(base_dir=root_path, in_dir_name=in_dir, out_dir_name=tar_dir)
# db_test = EM_dataset(base_dir=root_path, in_dir_name=in_dir, out_dir_name=tar_dir,
#                                transform=transforms.Compose(
#                                    [RandomGenerator(output_size=[img_size, img_size])]))
print("The length of train set is: {}".format(len(db_train)))
testloader = DataLoader(db_test, batch_size=6, shuffle=True, num_workers=0, pin_memory=True, drop_last = True)

model = ResUnet(1,[64, 128, 256, 512, 512]).to(device)
num_epochs = 3#50
steps_per_epoch = 2#100

train_model_new(trainloader, model,num_epochs,steps_per_epoch,dest_dir,device)

# best_model_cp = torch.load('output/best_model.pth')
# best_model_epoch = best_model_cp['epoch']
# print(f"Best model was saved at {best_model_epoch} epochs\n")
# model.load_state_dict(best_model_cp['model_state_dict'])

evaluate_model(testloader, model,dest_dir,device)




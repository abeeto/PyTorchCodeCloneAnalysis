# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:54:34 2022

@author: Admin
"""

import torch

device = torch.device('cpu') 
seed = 264

'''Batch Configs'''
chunk_size = 1024 * 32
batch_size = 64
num_pixel_batch = batch_size ** 2


'''Learning rate configs'''
LR = 5e-4
lr_decay = 250
decay_rate = 0.1
decay_steps = lr_decay * 1000


'''Data'''
path = '/content/drive/MyDrive/NeRF/car.npz'


'''Save paths'''
log_dir = '/content/drive/MyDrive/NeRF/model'
model_save_path = '/content/drive/MyDrive/NeRF/logs'


'''Volume Rendering Parameters'''
near_bound = 1.0
far_bound = 4.0

num_coarse_loc = 64
num_fine_loc = 128


'''Training parameters'''

num_iters = 300000
display_on_tensorboard = 100



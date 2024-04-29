#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  BaseNet class test
    Email : autuanliu@163.com
    Date：2018/04/05
"""
from models.BaseNet_class import BaseNet
from models.simpleNet import SimpleConv
from models.utils.get_data import train_val_test_spilt
from models.utils.utils_imports import *

# get data and configures
data_dir = PurePath('datasets/FashionMNIST')
tfs = {'train': transforms.ToTensor(), 'valid': transforms.ToTensor(), 'test': transforms.ToTensor()}
loaders, datasizes = train_val_test_spilt(
    data_dir, 'FashionMNIST', [64, 64, 64], tfs, [True, False], random_seed=250, valid_size=0.1, num_workers=0, pin_memory=False)

# model configure
net = SimpleConv()
opt = optim.Adam(net.parameters(), lr=1e-3)
configs = {
    'model': net,
    'opt': opt,
    'criterion': nn.CrossEntropyLoss(),
    'dataloaders': {
        'train': loaders[0],
        'valid': loaders[1],
        'test': loaders[2]
    },
    'data_sz': {
        'train': datasizes[0],
        'valid': datasizes[1],
        'test': datasizes[2]
    },
    'lrs_decay': lr_scheduler.StepLR(opt, step_size=50),
    'prt_freq': 5,
    'epochs': 150,
    'checkpoint': PurePath('logs/checkpoint'),
    'visual_dir': 'logs/visual',
    'prt_dir': PurePath('logs/prt/prt_log.txt')
}

# construct sub-model from BaseNet
sub_model = BaseNet(configs)
# train and test
sub_model.train_m()
sub_model.test_m()
sub_model.loss_acc_plot()
# sub_model.visualize()

# get property
# print(sub_model.res)
# print(sub_model.best_acc)
# print(sub_model.best_model)
# print(sub_model.res_model)
# print(sub_model.best_model_wts)
# print(sub_model)

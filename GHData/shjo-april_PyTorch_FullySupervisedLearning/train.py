# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import sys
import copy
import time
import glob
import json
import random
import pickle
import argparse

import numpy as np
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from core.networks import *
from core.data_utils import *

from core.utils import *
from utility.utils import *

parser = argparse.ArgumentParser()

###############################################################################
# GPU Config
###############################################################################
parser.add_argument('--use_gpu', default='0', type=str)

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--data_dir', default='./data/', type=str)
parser.add_argument('--use_cores', default=mp.cpu_count(), type=int)

###############################################################################
# Network
###############################################################################
parser.add_argument('--experiment_name', default='CIFAR-10_seed@0', type=str)
parser.add_argument('--dataset_name', default='CIFAR-10', type=str)

parser.add_argument('--weight_decay', default=1e-4, type=float)

###############################################################################
# Training
###############################################################################
parser.add_argument('--optimizer', default='SGD', type=str)
parser.add_argument('--learning_rate', default=0.1, type=float)

parser.add_argument('--image_size', default=32, type=int)
parser.add_argument('--batch_size', default=64, type=int)

parser.add_argument('--log_interval', default=100, type=int)
parser.add_argument('--val_interval', default=5000, type=int)
parser.add_argument('--max_iterations', default=100000, type=int)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.use_gpu

if __name__ == '__main__':
    # 1. Config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        args.batch_size *= len(args.use_gpu.split(','))

        if args.batch_size > 256:
            args.learning_rate *= args.batch_size / 256

    set_seed(args.seed)

    log_dir = create_directory(f'./experiments/logs/')
    model_dir = create_directory('./experiments/models/')
    tensorboard_dir = create_directory(f'./experiments/tensorboards/{args.experiment_name}/')

    train_csv_path = log_dir + f'{args.experiment_name}_train.csv'
    valid_csv_path = log_dir + f'{args.experiment_name}_validation.csv'
    log_path = log_dir + f'{args.experiment_name}_log.txt'
    model_path = model_dir + f'{args.experiment_name}.pth'
    
    log_func = lambda string='': log_print(string, log_path)
    
    if os.path.isfile(model_path): os.remove(model_path)
    if os.path.isfile(train_csv_path): os.remove(train_csv_path)
    if os.path.isfile(valid_csv_path): os.remove(valid_csv_path)
    if os.path.isfile(log_path): os.remove(log_path)
    
    # 2. Dataset
    if args.dataset_name == 'CIFAR-10':
        train_dataset, validation_dataset, test_dataset, in_channels, classes = get_CIFAR_10(args.data_dir, args.image_size)

    elif args.dataset_name == 'CIFAR-100':
        train_dataset, validation_dataset, test_dataset, in_channels, classes = get_CIFAR_100(args.data_dir, args.image_size)

    elif args.dataset_name == 'STL-10':
        train_dataset, validation_dataset, test_dataset, in_channels, classes = get_STL_10(args.data_dir, args.image_size)

    elif args.dataset_name == 'MNIST':
        train_dataset, validation_dataset, test_dataset, in_channels, classes = get_MNIST(args.data_dir, args.image_size)

    elif args.dataset_name == 'KMNIST':
        train_dataset, validation_dataset, test_dataset, in_channels, classes = get_KMNIST(args.data_dir, args.image_size)

    elif args.dataset_name == 'FashionMNIST':
        train_dataset, validation_dataset, test_dataset, in_channels, classes = get_FashionMNIST(args.data_dir, args.image_size)

    elif args.dataset_name == 'SVHN':
        train_dataset, validation_dataset, test_dataset, in_channels, classes = get_SVHN(args.data_dir, args.image_size)
    
    train_indices, validation_indices = split_train_and_validation_datasets(train_dataset, classes, ratio=0.1)
    
    log_func('# Dataset ({})'.format(args.dataset_name))
    log_func('[i] The size of train dataset = {}'.format(len(train_indices)))
    log_func('[i] The size of validation dataset = {}'.format(len(validation_indices)))
    log_func('[i] The size of test dataset = {}'.format(len(test_dataset)))
    log_func()
    
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.use_cores // 4, pin_memory=False, drop_last=True, sampler=train_sampler)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, num_workers=args.use_cores // 4, pin_memory=False, drop_last=False, sampler=validation_sampler)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.use_cores // 4, pin_memory=False, drop_last=False)
    
    # 3. Networks
    model = WideResNet(in_channels, classes)

    def accuracy_fn(logits, labels):
        condition = torch.argmax(logits, dim=1) == labels
        accuracy = torch.mean(condition.float())
        return accuracy * 100

    loss_fn = F.cross_entropy
    
    # 4. Optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.max_iterations // 3, gamma=0.1)

    # 5. Training
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        log_func('[i] Distributed Training  : {}, device count = {}, batch size = {}'.format(os.environ['CUDA_VISIBLE_DEVICES'], torch.cuda.device_count(), args.batch_size))

    model = model.to(device)
    
    train_timer = Timer()
    train_avg = Average_Meter(['loss', 'accuracy'])

    test_timer = Timer()
    test_avg = Average_Meter(['loss', 'accuracy'])

    best_valid_accuracy = -1

    csv_print(['Iteration', 'Learning_Rate', 'Loss', 'Accuracy', 'time(sec)'], train_csv_path)
    csv_print(['Iteration', 'Loss', 'Accuracy', 'Best Accuracy', 'time(sec)'], valid_csv_path)

    train_iter = iter(train_loader)

    for iteration in range(1, args.max_iterations + 1):

        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        loss = loss_fn(logits, labels)
        accuracy = accuracy_fn(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss = loss.item()
        accuracy = accuracy.item()

        train_avg.add({'loss':loss, 'accuracy':accuracy})
        
        if iteration % args.log_interval == 0:
            data = [iteration, get_learning_rate_from_optimizer(optimizer)] + train_avg.get(clear=True) + [train_timer.tok(clear=True)]

            log_func('[i] iter={}, lr={}, loss={:.4f}, accuracy={:.2f}%, train_sec={}sec'.format(*data))
            csv_print(data, train_csv_path)

        if iteration % args.val_interval == 0:
            model.eval()
            test_timer.tik()
            
            def get_loss_and_accuracy(test_loader):
                evaluation_iter = iter(test_loader)
                evaluation_length = len(test_loader)

                with torch.no_grad():
                    for step, (images, labels) in enumerate(evaluation_iter):
                        images = images.to(device)
                        labels = labels.to(device)
                        
                        logits = model(images)

                        loss = loss_fn(logits, labels)
                        accuracy = accuracy_fn(logits, labels)

                        test_avg.add({'loss' : loss.item(), 'accuracy' : accuracy.item()})

                        sys.stdout.write('\r# Evaluation = {:.02f}% [{}/{}]'.format((step + 1) / evaluation_length * 100, step + 1, evaluation_length))
                        sys.stdout.flush()
                print()
                
                loss, accuracy = test_avg.get(clear=True)
                return loss, accuracy

            loss, accuracy = get_loss_and_accuracy(validation_loader)

            if best_valid_accuracy == -1 or best_valid_accuracy < accuracy:
                best_valid_accuracy = accuracy
                save_model(model, model_path)

            data = [iteration, loss, accuracy, best_valid_accuracy, test_timer.tok(clear=True)]
            log_func('[i] iter={}, valid_loss={:.4f}, valid_accuracy={:.2f}%, best_valid_accuracy={:.2f}%, test_sec={}sec'.format(*data))
            csv_print(data, valid_csv_path)
            
            model.train()

    # final test
    load_model(model, model_path)

    _, test_accuracy = get_loss_and_accuracy(test_loader)
    log_func('# Final Test Accuracy = {:.2f}%'.format(test_accuracy))

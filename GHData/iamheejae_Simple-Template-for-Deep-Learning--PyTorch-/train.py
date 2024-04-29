import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import os
import argparse
import sys
import pdb
import time

sys.path.insert(0,'./models/')
sys.path.insert(0,'./utils/')

from mydataset import mydataset
from models import main_model
from solver_DataParallel import Solver

#******************** Get options for training********************#

parser = argparse.ArgumentParser(description = 'Argument for my network')

parser.add_argument('--train_set', metavar='T', type=str, default='train',
                    help='training dataset root.')
parser.add_argument('--val_set', metavar='V', type=str, default='validation',
                    help='validation dataset root.')
parser.add_argument('--batch_size', metavar='B', type=int, default=8,
                    help='batch size used for training. Default 8')
parser.add_argument('--learning_rate', metavar='L', type=float,
                    default=1e-5, help='learning rate. Default 1e-5')
parser.add_argument('--num_epochs', metavar='N', type=int, default=50,
                    help='number of training epochs. Default 100')
parser.add_argument('--fine_tune', dest='fine_tune', action='store_true',
                    help='fine tune the model under check_point dir,\
                    instead of training from scratch. Default False')
parser.add_argument('--verbose', dest='verbose', action='store_true',
                    help='print training information. Default False')
parser.add_argument('--optim',dest='optim',  type=str, default='Adam',
                    help='how to optimize')
parser.add_argument('--option',dest='option', type=str, default='random',
                    help='select option')
parser.add_argument('--gpus',dest='gpus',  type=int, default=2,
                    help='Number of GPU for DataParallel package in Pytroch')
parser.add_argument('--parallel', dest='parallel', action='store_true',
                    help='print use DataParallel or not. Default false')

args = parser.parse_args()

def get_full_path(dataset_path):
    """
    Get full path of data based on configs and target path
    example: datasets/train    """
    
    return os.path.join(dataset_path)

def display_config():
    print('##########################################################################')
    print('#             My Simple Template for Deep Learning - Pytorch             #')
    print('#                   by Heejae Kim (coolhj37@gmail.com)                   #')
    print('##########################################################################')
    print('')
    print('-------------------------------YOUR SETTINGS------------------------------')
    for arg in vars(args):
        print("%15s: %s" % (str(arg), str(getattr(args, arg))))
    print('')

def main():
    display_config()
    #Set root path for training and validation data
    train_root = get_full_path(args.train_set)
    val_root = get_full_path(args.val_set)
    #Set view mode: Select 2 views, one for target view and another for neighbor view
    option = args.option
    
    print('Contructing dataset...')
    train_dataset = mydataset(train_root, args.option)
    val_dataset = mydataset(val_root, args.option)    

    # Set Model
    model = main_model.main_model()

    #**********Optimiazion Method**********#
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum = 0.9, weight_decay = 0.0005)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas = (0.9,0.999), eps = 1e-08, weight_decay = 0.00005, amsgrad=False)   
    elif args.optim == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate, lr_decay=0, weight_decay=0.00005, initial_accumulator_value=0, eps=1e-10)   
    
    #************** Checkpoint **************#
    check_point = os.path.join('check_point')
    solver = Solver(
        model, check_point, batch_size=args.batch_size,
        num_epochs=args.num_epochs, learning_rate=args.learning_rate, optimizer=optimizer,
        fine_tune=args.fine_tune, verbose=args.verbose, gpus=args.gpus, parallel=args.parallel)

    print('Now Start Training ...')
    solver.train(train_dataset, val_dataset)


if __name__ == '__main__':
    main()


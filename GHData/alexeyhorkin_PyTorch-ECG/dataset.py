import os
import torch
import json
from torch.utils.data import  DataLoader
from utils import ECG_dataset


def get_dataset(args):
    ''' 
    Special func for return train and test dataloader 
    '''

    train_ecg_dataset = ECG_dataset(args.path_to_DataFile, is_train=True)
    test_ecg_dataset = ECG_dataset(args.path_to_DataFile, is_train=False)

    train_dataloader = DataLoader(dataset = train_ecg_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_dataloader = DataLoader(dataset = test_ecg_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    return train_dataloader, test_dataloader


import os
import sys
import torchvision.transforms as transforms
import numpy as np
import options
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
import time
import options
import random
import tqdm
import shutil

def set_data():
    
    opt = options.train_options()
    protocol =opt.protocol
    root = str(opt.dataroot)
    data_path_train = os.path.join(root, "training")
    data_path_valid = os.path.join(root, "validation")
    data_path_test = os.path.join(root,"testing")
    normal_class = 1
    splits = ['train', 'valid','test']
    drop_last_batch = {'train': True,'valid' : True , 'test': True}
    shuffle = {'train': True, 'valid' : True, 'test': True}
    batch_size = {'train': opt.batch,'valid': opt.batch , 'test': opt.batch}

    transform = transforms.Compose(
        [
            transforms.Resize(opt.isize),
            transforms.RandomCrop(opt.cropsize),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
            

        ]
    )

    dataset = {}
    dataset['train'] = torchvision.datasets.ImageFolder(root = data_path_train, transform=transform)
    dataset['valid'] = torchvision.datasets.ImageFolder(root = data_path_valid, transform=transform)
    dataset['test'] = torchvision.datasets.ImageFolder(root = data_path_test, transform= transform)
    dataloader = {m: DataLoader(dataset=dataset[m],
                                                    batch_size=batch_size[m],
                                                    shuffle = shuffle[m],
                                                     num_workers=opt.workers,
                                                     drop_last=drop_last_batch[m] ) for m in splits}
    print("data loading complete")
    return dataloader




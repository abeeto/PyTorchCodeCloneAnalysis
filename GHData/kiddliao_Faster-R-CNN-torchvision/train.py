import torch
import torchvision
import pycocotools
from datasets import *
import os
from torch.utils.data import DataLoader
from torchvision import transforms

data_path=os.path.join('..','..','REMOTE','datasets','coco_shape')
params ={
    'project_name': '',
    'train_set': 'train',
    'val_set': 'val',
    'test_set': 'val',
    'num_gpus': 0,
    'obj_list': ['rectangle', 'circle']
}
stat_txt_path='shape_stat.txt'

train_params = {
    'batch_size': 4,
    'shuffle': True,
    'drop_last': True,
    'collate_fn': collater,
    'num_workers': 0
}
val_params = {
    'batch_size': 4,
    'shuffle': False,
    'drop_last': True,
    'collate_fn': collater,
    'num_workers': 0
}

train_set = DIYDataset(
        path=os.path.join(data_path, params['project_name']),
        set_name=params['train_set'],
        mean_std_path=stat_txt_path,  #计算训练集的均值和方差
        cal_mean_std=False,
        transform=transforms.Compose([Normalizer(mean_std_path=stat_txt_path),
                                      Augmenter(), Resizer(416)]))

train_generator = DataLoader(train_set, **train_params)

val_set = DIYDataset(path=os.path.join(data_path, params['project_name']),
        set_name=params['val_set'],
        cal_mean_std=False,
        transform=transforms.Compose(
            [Normalizer(mean_std_path=stat_txt_path),
            Augmenter(), Resizer(416)]))

val_generator = DataLoader(val_set, **val_params)


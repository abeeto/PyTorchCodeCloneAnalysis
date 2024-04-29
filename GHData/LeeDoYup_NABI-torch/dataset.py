from __future__ import print_function
import os
import json
import pickle
import numpy as np
import utils
import torch
import random
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class FeatureDataset(Dataset):
    def __init__(self, 
                 name,
                 dataset_path='./data'):
        super(FeatureDataset, self).__init__()
        self.keys = ['vital', 'demo', 'gt']
        self.set_features(name, dataset_path)
        self.tensorize()
        
    def set_features(self, name, dataset_path):
        print('loading features from files')
        self.data = {}
        for key in self.keys:
            self.data[key] = np.load(os.path.join(dataset_path, name+'_'+key+'.npy'))
            if key == 'gt':
                self.data[key] = np.expand_dims(self.data[key], axis=-1)
    
    def tensorize(self):
        self.tensor = {}
        for key in self.keys:
            self.tensor[key] = torch.from_numpy(self.data[key])

    def __getitem__(self, index):
        vital = self.tensor['vital'][index]
        #demo = self.tensor['demo'][index]
        gt = self.tensor['gt'][index]
        return vital, gt

    def __len__(self):
        return len(self.data['gt'])
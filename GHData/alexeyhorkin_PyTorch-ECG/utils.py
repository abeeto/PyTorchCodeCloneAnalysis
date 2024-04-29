import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ECG_dataset(Dataset):
    def __init__(self, path_to_DataFile, is_train=True, transform=None):
        ''' 
        Class for custom dataset for ecg signals
        '''
        self.data_dict = json.load(open(os.path.join(path_to_DataFile), 'r') )
        self.path_to_DataFile = path_to_DataFile
        self.transform = transform
        self.procent_of_train = 80
        self.is_train = is_train

    def __len__(self):
        if self.is_train:
            return int(len(self.data_dict)*self.procent_of_train/100.0)
        else:
            return int(len(self.data_dict)*(1.0-self.procent_of_train/100.0))
    
    def __getitem__(self, idx):
        if not self.is_train:
            idx = self.__len__()-1 + idx
        idx = 104 # for experiment
        keys = list(self.data_dict.keys())
        leads = self.data_dict[keys[idx]]['Leads']
        signals = [ leads[i]['Signal'] for i in list(leads.keys())]
        signals = np.array(signals, dtype=np.float32)
        label = np.array(list(self.data_dict[keys[idx]]['StructuredDiagnosisDirect'].values())).astype(np.float32)
        sample = (signals, label)

        if self.transform:
            sample = self.transform(sample)
        return sample 


def apply_trashold(tensor, treshold):
    tmp = tensor.cpu().numpy()
    bstch_size, count_of_class = np.shape(tmp)
    for i in range(bstch_size):
        for j in range(count_of_class):
            if tmp[i][i]<=treshold:
                tmp[i][j] = 0.0
            else:
                tmp[i][j] = 1.0
    return tmp

def is_equal(out, target):
    tmp = np.sum((out - target)**2)
    if tmp>0:
        return False
    else:
        return True 
    
def get_accuracy(out_batch, taget_batch):
    acc = []
    for ind, out in enumerate(out_batch):
        acc.append(is_equal(out,taget_batch[ind].cpu().numpy()))
    acc = np.array(acc).astype(np.float32)
    return acc.sum(), len(taget_batch)
import os
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
import numpy as np
import random
import math
from utils import *
from config import *

#A dataset template
class NETDataset(data.Dataset):
    def __init__(self, data, label,index, need_aug):
        self.data = data
        self.label = label
        self.index = index
        self.need_aug = need_aug

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        if (self.need_aug):
            # scale
            scale = random.random() + 0.5
            if random.random() > 0.5:
                M1 = torch.tensor([[scale, 0], [0, scale]]).to(device) 
            else:
                M1 = torch.tensor([[-scale, 0], [0, -scale]]).to(device) 
            data = torch.mm(data, M1)

            # rotate
            theta = 2 * math.pi * random.random() - math.pi
            M2 = torch.tensor([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]]).to(device)
            data = torch.mm(data,M2)

        label = self.label[idx]
        if(opt.neighbor_type == 'knn'):
            index = 0
        else:
            index = self.index[idx]
        return data, label, index


import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

 transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
          transforms.Normalize(
        mean=[0.5],
        std=[0.5])])

evens = list(range(0, len(train_data), 20))
odds=list(range(0,len(test_data),20))
train_data_lim=torch.utils.data.Subset(train_data,evens)
test_data_lim=torch.utils.data.Subset(test_data,odds)
num_train=len(train_data_lim)
indices = list(range(num_train))
split = int(np.floor(0.2* num_train))
np.random.shuffle(indices)
train_idx,valid_idx=indices[split:],indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
train_loader=DataLoader(train_data_lim,batch_size=64,sampler=train_sampler)
valid_loader=DataLoader(train_data_lim,batch_size=64,sampler=valid_sampler)
test_loader=DataLoader(test_data_lim,batch_size=64)


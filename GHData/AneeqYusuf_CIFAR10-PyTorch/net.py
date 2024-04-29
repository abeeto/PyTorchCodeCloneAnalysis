# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import pandas as pd
import torch.utils.data as data
import cv2
import os

classes = {'upright': 0, 'rotated_left': 1, 'rotated_right': 2, 'upside_down': 3}

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2704, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 4)
    
    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = x.view(-1, 2704)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = func.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class data_load(data.Dataset):
    
    def __init__(self, transform, folder, truth_file=None):
        self.transform=transform
        self.truth_file = truth_file
        self.image_list = []
        self.image_label = []
        if (truth_file == None):
            imgs = os.listdir(folder)
            for i in range (len(imgs)):
                filename = folder + imgs[i]
                self.image_list.append(cv2.imread(filename))
            self.X=np.array(self.image_list)
        else:
            self.df=pd.read_csv(truth_file)
        #changed label location to -1
            for i in range (len(self.df)):
                filename = folder + self.df.iloc[i]['fn']
                self.image_list.append(cv2.imread(filename))
                self.image_label.append(classes.get(self.df.iloc[i]['label']))
            self.X=np.array(self.image_list)
            self.Y=np.array(self.image_label, dtype=np.int)

    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, index):
        if (self.truth_file != None):
            label=self.Y[index]
            item=self.X[index]
            if (self.transform):
                item=self.transform(item)
            return (item, label)
        else:
            item=self.X[index]
            if (self.transform):
                item=self.transform(item)
            return (item)

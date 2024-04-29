import torch
import torchvision
import torch.nn  as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
import os
import glob
import shutil
import random
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import copy

class ImageTempDataset(torch.utils.data.Dataset):
    classes = ['0', '1', '2']
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    def __init__(self, csv_path, img_path, transform=data_transforms):
        self.img_path = img_path
        self.transform = transform
        df = pd.read_csv(csv_path, index_col=0)
        self.images = list(df['img_path'])
        self.temps = list(df['tem'])
        self.labels = list(df['class'])
        
    def __getitem__(self, index):
        image = self.images[index]
        image = os.path.join(self.img_path, image)
        with open(image, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        temp = self.temps[index]
        label = self.labels[index]
        return image, temp, label
    
    def __len__(self):
        return len(self.images)
        

class CustomDataset(torch.utils.data.Dataset): 
    classes = ['0', '1', '2']
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    def __init__(self, path_lst, img_path, transform=data_transforms):
        '''
        path_lst:
        -	class0_under.csv
        -	class1_under.csv
        -	class2_under.csv
        img_path: 画像が保存されているパス
        '''
        self.img_path = img_path
        self.transform = transform
        self.images = []
        self.labels = []
        df_0 = pd.read_csv(path_lst[0], index_col=0)
        df_1 = pd.read_csv(path_lst[1], index_col=0)
        df_2 = pd.read_csv(path_lst[2], index_col=0)
        self.images = list(df_0['img_path']) + list(df_1['img_path']) +  list(df_2['img_path'])
        self.labels = list(df_0['class0'] * 0) + list(df_1['class1'] * 1) + list(df_2['class2'] * 2) 

    def __getitem__(self, index):
        image = self.images[index]
        image = os.path.join(self.img_path, image)
        label = self.labels[index]
        with open(image, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.images)
import os
import numpy
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from utils import data_names_create,file_to_PIL

class M2Det_320Dataset(Dataset):
    '''
    # Folder Setup:
    # imgs/ lbls/ data.names

    '''
    def __init__(self,path_to_dataset,transforms = None,target_transforms = None):
        # PATHS
        file_name = 'data.names'
        full_path = os.path.join(path_to_dataset,file_name)
        # Transforms
        self.transforms = transforms
        self.target_transforms = target_transforms

        self.data_list = []
        self.target_list = []
        # Read Lines
        with open(full_path,'r') as reader:
            lines = reader.readlines()
            for line in lines:
                self.data_list.append(line.rstrip('\n'))

    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, index):
        file_name = self.data_list[index]
        img,target = file_to_PIL(file_name,genereate_lbls=True)

        if self.transforms is not None:
            img = self.transforms(img)
        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return img,target
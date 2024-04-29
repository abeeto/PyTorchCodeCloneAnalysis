import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import os

from utils import *

# Dataset
class ImageDataset(torch.utils.data.Dataset):
        def __init__(self,num_examples=10,path=None,transforms=None):
                # TODO
                # 1. Initialize file paths or a list of file names. 
           self.transforms = transforms
           self.path = path
           self.image_names = os.listdir(self.path)[:num_examples]
           self.img_loader = image_load_function

        def __getitem__(self, index):
                if self.transforms is not None:
                        return self.transforms(self.img_loader(self.path + self.image_names[index]))
                else:
                        return self.img_loader(self.path + self.image_names[index])

        def __len__(self):
                # You should change 0 to the total size of your dataset.
                return len(self.image_names)

class MyDataset(torch.utils.data.Dataset):
        def __init__(self,dataset_A,dataset_B):
                self.dataset_A = dataset_A
                self.dataset_B = dataset_B

        def __getitem__(self, index):
                return self.dataset_A[index],self.dataset_B[index]

        def __len__(self):
                return min(len(self.dataset_A),len(self.dataset_B))
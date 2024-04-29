import torch 
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.transforms import CenterCrop, Compose, Resize, ToPILImage, ToTensor, Normalize
from torch import long, tensor
from torch.utils.data.dataset import Dataset


class MaskDetectionDataset(Dataset):
    """Mask Detection Dataset"""
    def __init__ (self, dataFrame):
        self.dataFrame = dataFrame
        self.transformations = Compose([ToPILImage(),Resize(256),CenterCrop(224),ToTensor(),Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
     
         
    def __getitem__(self, key):
        row = self.dataFrame.iloc[key]
        image = cv2.imdecode(np.fromfile(row['image'], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        return{
            'image': self.transformations(image),
            'mask': tensor([row['mask']], dtype=long),
        }
    
    def __len__(self):
        return len(self.dataFrame.index)



     
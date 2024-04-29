# -*- coding: utf-8 -*-
"""
@author: roshan
"""

import numpy as np
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torch
import random
from PIL import Image
import csv
import PIL
import imageio

import dataAugmentationTransforms as dat

  
class fundusData(Dataset):

    def __init__(self, csvFile, augmentation = True):
        """
        Args:            
            csv (string): path to the folder where images are and their respective labels
            augmentation: True for training, False for validation and testing
            
            
            output: image tensor and labels
        """
        self.imageTrain_transform = transforms.Compose([
            transforms.Resize((224, 224), 3),
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.csvFile = csvFile
        with open(self.csvFile, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            self.dataInfo = list(csv_reader)        
        
        self.augmentation = augmentation
        if len(self.dataInfo) == 0:
            raise Exception('No images/labels are found in {}'.format(self.root))

    def __getitem__(self, index):
        image = imageio.imread(self.dataInfo[index][1]) # load images
        label = np.array(np.int64(self.dataInfo[index][2])) # load labels
        
        imgPil = Image.fromarray(image)       
        
        if self.augmentation:
            #Augmentation
            # 0: no augmentation, 1: flip, 2: rotate, 3: gamma, 4: translate
            aug_mthd = random.randint(0,4)
            
            if aug_mthd == 1:
                # flip {0: vertical, 1: horizontal, 2: both}
                option = random.randint(0, 2)
                imgPil = dat.flip(imgPil, option)
            
            if aug_mthd == 2:
                #rotate {0: NO rotation, 1: YES rotation}
                option = 1
                angle = random.randint(-180, 180)
                imgPil = dat.rotate(imgPil, angle, option, PIL.Image.BILINEAR)
            
            if aug_mthd == 3:
                #change gamma {0: NO, 1: YES}
                option = 1
                factor = random.uniform(0.75, 1.25)
                imgPil = dat.gammaCorrection(imgPil, factor, option)
            
            if aug_mthd == 4:
                #translate {0: NO, 1: YES}
                option = 1
                x = random.randint(1,5)
                y = random.randint(1,5)
                imgPil = dat.translate(imgPil, x, y, option)
        
        # resize image and label, then convert to tensor.
        imageTensor = self.imageTrain_transform(imgPil)
        label = torch.from_numpy(label)
            
        return (imageTensor, label)

    def __len__(self):
        return len(self.dataInfo)




def loader(dataset, batch_size, num_workers=0, shuffle=True):
    input_images = dataset
    input_loader = torch.utils.data.DataLoader(dataset=input_images, batch_size=batch_size,
      shuffle=shuffle,
      num_workers=num_workers)
    return input_loader
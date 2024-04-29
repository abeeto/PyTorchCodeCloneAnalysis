import numpy as np
import monai
from monai.data import ImageDataset
from monai.transforms import *
from pathlib import Path
import os
import pandas as pd
import numpy as np

##define dataset (through kinda wrapping over the ImageDataset)
def UKB_T1(data_path, split, transform, **kwargs):
    """주의 : train/test split rate : 0.75로 preset (희환쌤 코드 처럼 분리할 수도 있을 것 같으나, 이것은 나중에 하자 )"""
    
    imgs = sorted([f for f in Path(data_path).iterdir()])

    #TRAIN_TEST SPLIT (take first 75% or last 25% of iamge list depending on split)
    imgs = imgs[:int(len(imgs)*0.75)] if split == "train" else imgs[int(len(imgs)*0.75): ] #i.e. 앞의 75% if train/뒤의 25% if test
                        #list of subject directories (PosixPath) to use
    
    


    lbls = np.zeros(len(imgs))
    lbls = lbls.astype(int)

    return ImageDataset(image_files=imgs, labels = lbls, transform = transform)


class Transform_yAware:
    def __init__(self):
        
        self.transform = monai.transforms.Compose([
            #normalize , flip, blur, noise, cutout, crop
            ScaleIntensity(), AddChannel(),
            RandFlip(prob = 0.5),
            RandGaussianSharpen(sigma1_x=(0.1, 1.0), sigma1_y=(0.1, 1.0), sigma1_z=(0.1, 1.0), sigma2_x=0.1, sigma2_y=0.1, sigma2_z=0.1,prob=0.5),
            ResizeWithPadOrCrop(spatial_size =  (182,218,182), method = "symmetric", mode = "constant"),
            NormalizeIntensity()
            ,ToTensor()
        ])
        
   
        #어디에 뭐가 들어있는지 확인하기 위해서, (182,20,182)로 함
        self.transform_prime = monai.transforms.Compose([
            #normalize , flip, blur, noise, cutout, crop
            ScaleIntensity(), AddChannel(),
            RandFlip(prob = 0.5),
            RandGaussianSharpen(sigma1_x=(0.1, 1.0), sigma1_y=(0.1, 1.0), sigma1_z=(0.1, 1.0), sigma2_x=0.1, sigma2_y=0.1, sigma2_z=0.1,prob=0.5),
            ResizeWithPadOrCrop(spatial_size =  (182,218,182), method = "symmetric", mode = "constant"),
            NormalizeIntensity()
            ,ToTensor()
        ])
        
         
            
    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1,y2    


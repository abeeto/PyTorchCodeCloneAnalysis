# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 23:13:27 2020

@author: juika
"""


import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.utils


from skimage import exposure
from skimage.transform import resize

import pandas as pd


from PIL import Image




    
    
"""
REFUGE dateset with 800 RIM masks based on fundus images
80 glaucoma cases; 720 normal cases

Setting for
data_folder = "Z:\\GarvinLabDL_Data\\REFUGE\\Mix\\data"
"""
class DatasetREFUGE(Dataset):
    def __init__(self, data_df, input_img_range, output_img_size):    
        
        self.image_list = []
        self.output_img_size = output_img_size
        self.input_img_rescale_range = input_img_range

        for idx, row in data_df.iterrows():
            label = row["Label"]
            image_path = row["Path"]
            info_dict = {"label":label, "image_path":image_path}
            self.image_list.append(info_dict)

    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        image_file = self.image_list[idx]["image_path"]
        numpy_img = plt.imread(image_file)
        ## plt.imread --> automatically convert image to array with range [0 - 1]
            
        # resize the images
        output_height, output_width = self.output_img_size
        new_img = resize(numpy_img, (output_height, output_width), anti_aliasing=False)

        # rethreshold the image
        input_min, input_max = self.input_img_rescale_range
        mid_value = (input_min+input_max)*0.5
        for h in range(output_height):
            for w in range(output_width):
                p = new_img[h,w]
                th = 0.15*(input_max-input_min)
                if p >= input_max-th:
                    new_img[h,w] = input_max
                elif p <= input_min + th:
                    new_img[h,w] = input_min
                else:
                    new_img[h,w] = mid_value


        numpy_img = np.expand_dims(new_img, axis=2) # add the color channel = 1
        # Transpose the array shape from numpy to tensor format
        # numpy image: H x W x C
        # torch image: C X H X W
        tensor_format_img = numpy_img.transpose((2, 0, 1))
        tensor_img = torch.from_numpy(tensor_format_img).type(torch.float32)    
    
        label = self.image_list[idx]["label"]

        sample = {"image":tensor_img, "label":label}
        return sample
        
    
    
    

# data_folder = "Z:\\GarvinLabDL_Data\\REFUGE\\Mix\\data"    
# test_img = plt.imread(os.path.join(data_folder, "t_g0015_Cropped_Disc.png"))
    
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)  
# ax.imshow(test_img)    
    
       
# new_img = resize(test_img, (112, 112), anti_aliasing=False)
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)  
# ax.imshow(new_img) 
    
    
    
    
    

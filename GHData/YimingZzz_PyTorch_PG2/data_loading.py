import math
import sys
import random
import os
import json
import torch
import numpy as np
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class DeepFashionDataset(Dataset):
    def __init__(self, file_list, root_dir = '/home/yiming/code/data/DeepFashion/DF_img_pose/', key_points = 18, mode = "train"):
        self.root_dir = root_dir
        self.file_list = file_list
        self.key_points = key_points 
        self.mode = mode

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        source_img_name = self.file_list[idx][0].split('.')[0]
        target_img_name = self.file_list[idx][1].split('.')[0]
        target_pose_list = []

        if self.mode == "train":
        #get the path
            if 'Flip' in self.file_list[idx][0]: 
                img_dir = os.path.join(self.root_dir, 'train_img_flip/')    
                heatmap_dir = os.path.join(self.root_dir, 'train_img_flip_heatmap/')
                mask_dir = os.path.join(self.root_dir, 'train_img_flip_mask/')
            else:
                img_dir = os.path.join(self.root_dir, 'train_img/')    
                heatmap_dir = os.path.join(self.root_dir, 'train_img_heatmap/')
                mask_dir = os.path.join(self.root_dir, 'train_img_mask/')   
        else:
            img_dir = os.path.join(self.root_dir, 'test_samples_img/')
            heatmap_dir = os.path.join(self.root_dir, 'test_samples_img_heatmap/')
            mask_dir = os.path.join(self.root_dir, 'test_samples_img_mask/')


        #get the source and target image
        source_img = io.imread(os.path.join(img_dir, (source_img_name + '.jpg')))
        target_img = io.imread(os.path.join(img_dir, (target_img_name + '.jpg')))                

        #normalization and get the Tensor format
        source_img = source_img.transpose([2, 0, 1])
        source_img = source_img/127.5 - 1
        source_img = torch.from_numpy(source_img).float()
        target_img = target_img.transpose([2, 0, 1])
        target_img = target_img/127.5 - 1
        target_img = torch.from_numpy(target_img).float()

        #get the pose heatmap
        for i in range(self.key_points):
            target_pose_heatmap = io.imread(os.path.join(heatmap_dir, (target_img_name + '_' + 'heatmap' + '_' + str(i+1) +'.jpg')))
            target_pose_heatmap = target_pose_heatmap / 127.5 - 1
            target_pose_heatmap = target_pose_heatmap.reshape([1, 256, 256])
            target_pose_heatmap = torch.from_numpy(target_pose_heatmap).float()
            target_pose_list.append(target_pose_heatmap)    
        
        # concat 18 heatmaps together as input  
        for i in range(self.key_points):
            if i == 0:
                target_pose = target_pose_list[i]
            else:
                target_pose = torch.cat((target_pose, target_pose_list[i]), 0)
                
        #get the mask
        pose_mask = io.imread(os.path.join(mask_dir, (target_img_name + '_' + 'mask' + '.jpg')))
        pose_mask = pose_mask / 255
        pose_mask = torch.from_numpy(pose_mask).float()

        #one sample in the dataset(source_img, target_img, target_pose_heatmap, target_pose_mask)
        sample = {'source_img': source_img, 'target_img': target_img, 
                  'target_pose': target_pose, 'pose_mask': pose_mask}    
        return sample

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:45:18 2019

@author: Oleksiy Grechnyev
"""

import numpy as np
import pandas as pd
import torch
import os

from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

print('torch.__version__=', torch.__version__)

landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:,0], landmarks[:,1], s=10, marker='.', c='r')

if False:
    n=65
    img_name=landmarks_frame.iloc[n, 0]
    landmarks=landmarks_frame.iloc[n,1:].values
    landmarks=landmarks.astype('float').reshape(-1,2)
    
    print('Image name: {}'.format(img_name))
    print('Landmarks shape: {}'.format(landmarks.shape))
    print('First 4 Landmarks: {}'.format(landmarks[:4]))

    show_landmarks(io.imread('data/faces/'+img_name), landmarks)
    plt.show()
################################################################################
class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].values.astype('float').reshape(-1, 2)
        sample = {'image' : image, 'landmarks': landmarks}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
################################################################################        
if False:        
    face_dataset = FaceLandmarksDataset('data/faces/face_landmarks.csv','data/faces')
    if False:
        n_i = 4
        for i in range(n_i):
            plt.subplot(1, n_i, i+1)
            sample = face_dataset[i]
            plt.axis('off')
            plt.title('Sample # {}'.format(i))
            show_landmarks(**sample)
        plt.show()
################################################################################
    
class Rescale:
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h /w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w/h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks * [new_w /w, new_h/h]
        return {'image' : img, 'landmarks' : landmarks}
################################################################################
class RandomCrop:
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size)==2
            self.output_size = output_size
            
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        image = image[top:top+new_h, left:left+new_w]
        landmarks = landmarks - [left, top]
        return {'image' : image, 'landmarks' : landmarks} 
################################################################################
class ToTensor:
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = image.transpose((2, 0, 1))
        return {'image' : torch.from_numpy(image), 'landmarks': landmarks}
################################################################################
if False:
    scale = Rescale(256)
    crop = RandomCrop(128)
    composed = transforms.Compose([Rescale(256), RandomCrop(224)])
    # Apply transforms
    sample = face_dataset[65]
    for i, tsfrm in enumerate([scale, crop, composed]):
        transformed_sample = tsfrm(sample)
        plt.subplot(1, 3, i+1)
        plt.title(type(tsfrm).__name__)
        show_landmarks(**transformed_sample)

    plt.show()
################################################################################
transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                           root_dir='data/faces/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor(),
                                               ])
                                           )
#for i in range(5):
#    sample = transformed_dataset[i]
#    print(i, sample['image'].shape, sample['landmarks'].shape)
################################################################################

# Create a dataset loader
dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)

# Helper to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                        landmarks_batch[i, :, 1].numpy() + grid_border_size,
                        s=10, marker='.', c='r')

for i_batch, sample_batched in enumerate(dataloader):
    plt.figure()
    plt.axis('off')
    show_landmarks_batch(sample_batched)
    if i_batch==3:
        break
plt.show()

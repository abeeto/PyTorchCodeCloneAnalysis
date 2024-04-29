import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torchvision import transforms, utils

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform

""" 
    PyTorch image dataset object used in training
    
        This class will train a CNN to first transform/filter an image into game pieces
        and blank space white - object/game piece, black - no object/game piece detected.
        The result is a tensor with values 0 (black) and 1 (white)
        
        After isolating the game pieces, a feedforward neural network would be trained to
        identify and draw contors on the game pices (white blots of tensor)
        
        The trained model will then be tested until a certain accuracy threshold has been
        reached and then saved
        
"""

class image_contour_set (Dataset):
    
    def __init__(self, csv_filepath, root_dir, transform=None):
        # csv_filepath = file path to the csv file with all of the images
        # root_dir (string) = directory with all of the images
        self.frames = pd.read_csv(csv_filepath)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        # Return the number of frames there are in the frames list
        return len(self.frames)
    
    def __getitem__(self, index):
        
        image = os.path.join(self.root_dir, self.frames.iloc[index, 0])
        contours = self.frames.iloc[index, 1].as_matrix()
        
        sample = {'image': image, 'contours': contours}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

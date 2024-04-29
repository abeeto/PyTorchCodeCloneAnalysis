# -*- coding: utf-8 -*-
"""
Created on August 2022

Data as PyTorch type.

@authors: Nuria Gómez-Vargas
"""

import torch
from torch.utils.data import Dataset
from aux_funcs import l1
import random


##########################################################################################

# SEED

seed = 16
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

##########################################################################################


class myDataSet(Dataset):
    """
    Map-style dataset. https://pytorch.org/docs/stable/data.html
    A map-style dataset is one that implements the __getitem__() and __len__() protocols, 
    and represents a map from (possibly non-integral) indices/keys to data samples. 
    
    Parameters
    ----------
    X: torch.Tensor
        Explicative variables.
    y: torch.Tensor
        Response variable.
    num_features: int
        Number of features.
    """

    def __init__(self, dictionary, device, input_size, batch_size, list_rays):
        """
        It instantiates the data for PyTorch.
        
        Parameters
        ----------
        dictionary: dict
            Train, validation or test images.
        device: str
            CPU or GPU device.
        input_size: int
            Size of the vector of features.
        batch_size: int
            Size of the training batch.
        list_rays: list
            List with the ID of the individuals.
        """

        self.X, self.Y = torch.tensor([], device = device), torch.tensor([], device = device)
        i = 0

        for _ in range(int(batch_size/2)):

            ray1, ray2 = random.choices(list_rays, k = 2)
            [feature10, feature11] = random.choices(dictionary[ray1], k = 2)
            [feature2] = random.choices(dictionary[ray2], k = 1)
            
            i +=1
            self.X = torch.reshape( torch.cat((self.X, l1(feature10, feature11, device).unsqueeze(0)), 0), (i, input_size))
            self.Y = torch.cat((self.Y, torch.tensor([1.0], device = device)), 0)

            i +=1
            self.X = torch.reshape( torch.cat((self.X, l1(feature10, feature2, device).unsqueeze(0)), 0), (i, input_size))
            self.Y = torch.cat((self.Y, torch.tensor([0.0], device = device)), 0)

        
        self.num_features = self.X.size()[1]
        

    def __len__(self):
        return self.X.size()[0]
    

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

"""
@author: Federico Ottomano
"""

from torch.utils.data import Dataset
import numpy as np
import torch

class binded_dataset(Dataset):
    
    def __init__(self, df):
                
        self.ratios = df.values
        
        self.masks = np.where(self.ratios > 0, 1, 0.0)
        
    def __getitem__(self, idx):
        
        return (torch.tensor(self.masks[idx], dtype=torch.float32),
               torch.tensor(self.ratios[idx], dtype=torch.float32))
    
    def __len__(self):
        
        return len(self.ratios)
        
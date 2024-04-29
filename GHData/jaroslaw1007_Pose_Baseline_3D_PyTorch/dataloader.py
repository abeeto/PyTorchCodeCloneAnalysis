import numpy as np
import os
import torch
import glob
from torch.utils.data import Dataset, DataLoader

from cameras import *
from config import *
from utils import *
from read_utils import *

class Human36m(Dataset):
    def __init__(self, data_path, actions, rcams, training=True):
        self.use_sh = use_sh
        self.training = training
        
        # max length = 6343
        if self.training:
            self.train_input, self.train_output = None, None
            
            self.train_x = read_2dgt_data(data_path, actions, rcams, self.training)
            self.train_y = read_3d_data(data_path, actions, rcams, self.training)

            self.train_key_list = list(self.train_x.keys())        
            self._process_data()
        else:
            self.test_input, self.test_output = None, None

            self.test_x = read_2dgt_data(data_path, actions, rcams, self.training)
            self.test_y = read_3d_data(data_path, actions, rcams, self.training)

            self.test_key_list = list(self.test_x.keys())      
            self._process_data()
        
    def __len__(self):   
        if self.training:
            return self.train_input.shape[0]
        else:
            return self.test_input.shape[0]
        
    def __getitem__(self, idx):
        if self.training:
            inputs = torch.from_numpy(self.train_input[idx]).float()
            outputs = torch.from_numpy(self.train_output[idx]).float()
        else:
            inputs = torch.from_numpy(self.test_input[idx]).float()
            outputs = torch.from_numpy(self.test_output[idx]).float()
            
        return inputs, outputs
    
    def _process_data(self):
        if self.training:
            for i in range(len(self.train_key_list)):
                if i == 0:
                    self.train_input = self.train_x[self.train_key_list[i]]
                    self.train_output = self.train_y[self.train_key_list[i]]
                else:
                    self.train_input = np.vstack((self.train_input, self.train_x[self.train_key_list[i]]))
                    self.train_output = np.vstack((self.train_output, self.train_y[self.train_key_list[i]]))
        else:
            for i in range(len(self.test_key_list)):
                if i == 0:
                    self.test_input = self.test_x[self.test_key_list[i]]
                    self.test_output = self.test_y[self.test_key_list[i]]
                else:
                    self.test_input = np.vstack((self.test_input, self.test_x[self.test_key_list[i]]))
                    self.test_output = np.vstack((self.test_output, self.test_y[self.test_key_list[i]]))
                    
if __name__ == '__main__':
    rcams = cameras.load_cameras()
    custom_dataset = Human36m(data_path=BASE_PATH, actions=ACTIONS, rcams=rcams, training=True)
    custom_dataloader = DataLoader(dataset=custom_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
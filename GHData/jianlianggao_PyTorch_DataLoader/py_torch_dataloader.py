# This code is for loading data from different paths using PyTorch data Loader 
# NiFTI image files in this example
# by Jianliang Gao 31 Jan 2019
# 
#
# It is made available under the MIT License

from __future__ import print_function
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import sys,re,os,math
import numpy as np
import torch
import nibabel as nib

batch_size=2

class dHCPt2Dataset(Dataset):
   def __init__(self):        
        # This data set has only two columns. 
		# The 1st column contains the paths to the image files.
		# The 2nd column contains the labels (integer).
        imageinfo=np.loadtxt('./ML/dataset.csv',delimiter=',',dtype='str')
        self.len=imageinfo.shape[0]
        self.x_data=imageinfo[:,0]
        self.y_data=imageinfo[:,1].astype('int')
   def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
   def __len__(self):
        return self.len

if __name__ == "__main__":
    dataset = dHCPt2Dataset()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#    epoch_size=int(dataset.len**(1/float(batch_size)))
    # Use 1 epoch to walk through the list of image file paths.
    for epoch in range(1):
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            idx_range=batch_size
            print(epoch, i, 'inputs', inputs, "labels", labels)
            # Load one image file a time into memory, otherwise crash due to memory overflow.
            for j in range(len(inputs)):
               t2img=nib.load(inputs[j])
               t2img=t2img.get_fdata()
               print('image dims',t2img.shape)
               # do something with the loaded image.....
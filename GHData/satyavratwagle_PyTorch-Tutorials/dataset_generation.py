import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

'''
Assumptions :
    - Data is saved as individual datapoints in a .npy file
    - Each datapoint is labelled as 'sample_N.npy' where N is an integer 
    between 0 and #datapoints
    - All labels are saved in one .npy file named 'labels.npy'
'''

# Define Hyperparameters

batch_size = 100 # Change as you wish

'''
Description:
    - Creates a class MyDataset which provides a template to create custom 
    datasets from .npy and .csv files. (.npy used in example)
    - Requires a consistent file structure for label and data files.
    - Provides an iterator-like function to produce chunks of data when the 
    network needs it
    
Parameters:
    - labels_path : Path to a .npy file containing all labels in order
    - root_dir : Path containing all datapoint files as .npy   
    
Process :
    - All labels are loaded and length of labels is measured to establish 
    iterator end point.
    - One sample is imported from sample directory and preprocessed as 
    necessary before merging sample and corresponding label into one tuple
    - Tuple is returned
'''

class MyDataset(Dataset):

    def __init__(self, labels_path, root_dir):

        self.labels = np.load(labels_path)
        self.root_directory = root_dir

    def __len__(self):
        return len(self.labels)

    # Provides an iterator interface to the MyDataset object
    
    def __getitem__(self, idx):
        
        current_sample_path = os.getcwd()+self.root_dir+"sample_"+str(idx)+".npy"
        
        # Import labels and data and convert to tensor
        
        sample = torch.from_numpy(np.array([np.load(current_sample_path)])).type(torch.FloatTensor)
        label = torch.from_numpy(np.array(self.speakers[idx])).type(torch.LongTensor)
        
        # Insert data preprocessing here, if any.
        
        # Convert to tuple
        
        sample = {'sample':sample,'label':label}    
        return sample

'''
An instance of a MyDataset class with required parameters
'''

dataset_instance = MyDataset(speakers_path = os.getcwd()+"/path/to/labels.npy"
                                     ,root_dir = "/path/to/datapoint/directory")


'''
Uses the Dataloader API to access batches of data at once

Parameters:
    - Dataset object to access data from
    - batch_size : Batch Size of samples to take at once. (Network must be 
    configured to handled the number of samples)
    - shuffle : Boolean for random selection of samples from dataset
'''

dataset_generator = DataLoader(dataset_instance,
                               batch_size=batch_size,
                               shuffle=True)
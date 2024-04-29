from torch_geometric.data import DataLoader 
from MyOwnDataset import MyOwnDataset
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
def split_dataset(device,my_path,batch_size,validation_split,shuffle_dataset=True,random_seed=42):
    dataset = MyOwnDataset(my_path)
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset.data=dataset.data.to(device)

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size,sampler=valid_sampler)
    
    return {'train':train_loader,'val':validation_loader}
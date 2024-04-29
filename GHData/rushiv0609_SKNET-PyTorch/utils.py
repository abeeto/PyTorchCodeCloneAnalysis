import os
import zipfile
import gdown
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def download_data():
    '''
    Downlaod Data if not already downloaded
    '''
    
    dir_name = 'tiny-imagenet-200'
    zip_file = 'tiny-imagenet-200.zip'
    
    if not os.path.isdir(dir_name): # if directory not present then download and unzip
        if not os.path.exists(zip_file):
            url = 'https://drive.google.com/uc?id=1n-jwJulLoPraTe7KImctFjhsvufi_6yq'
            gdown.download(url, output=zip_file, quiet=False)
        
        print("Extracting ...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Extracting complete")

def get_dataloaders(batch_size = 256):
    '''
    Get Dataloaders with specified batch_size
    '''
    transform = transforms.Compose(
        [transforms.RandomCrop((56,56)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),])
    
    train_dir = 'tiny-imagenet-200/train'
    test_dir = 'tiny-imagenet-200/val'
    train_ds = torchvision.datasets.ImageFolder(train_dir, transform = transform)
    val_ds = torchvision.datasets.ImageFolder(test_dir, transform = transform)
    # train_ds, val_ds, _ = torch.utils.data.random_split(train_ds, [10, 10, len(train_ds)-20])
    
    
    print("Length of train, valid set: ",(len(train_ds), len(val_ds)))
    train_loader = DataLoader(train_ds, shuffle=True, batch_size= batch_size)
    val_loader = DataLoader(val_ds, shuffle=True, batch_size= batch_size)
    return train_loader, val_loader

def get_mean_std(loader):
    '''
    calculate mean and std. deviation for a dataset
    
    loader : torch DataLoader for dataset

    '''

    ch_sum, ch_sq_sum, num_batch = 0, 0, 0

    for data,_ in loader:
        ch_sum += torch.mean(data, dim = [0,2,3])
        ch_sq_sum += torch.mean(data**2, dim = [0,2,3])
        num_batch += 1

    mean = ch_sum / num_batch
    std = ((ch_sq_sum/num_batch) - (mean**2))**0.5

    return mean, std
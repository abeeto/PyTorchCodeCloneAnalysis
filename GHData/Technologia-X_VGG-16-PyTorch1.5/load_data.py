'''
Loads the given dataset using PyTorch Dataset module to be used with PyTorch's DataLoader Module for training/evaluation.
'''
import torch
from torch.utils.data import Dataset
from utils import generate_training_data


class LoadDataset(Dataset):
    '''
    Contains overwritten methods from Torch's Dataset class.
    '''

    def __init__(self, resized_image_size, total_images, classes, data_list, transform=None):
        '''
        Initiliaze dataset related parameters.
        '''
        self.resized_image_size = resized_image_size
        self.total_images = total_images
        self.classes = classes
        self.list_data = data_list
        self.transform = transform

    def __len__(self):
        '''
        Abstract method. Returns the total number of images.
        '''
        return self.total_images

    def __getitem__(self, idx):
        '''
        Abstract method. returns the image and label for a single input at index 'idx'.
        '''

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, label = generate_training_data(data=self.list_data[idx], resized_image_size=self.resized_image_size)

        sample = {
            'image':image,
            'label':label
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import cv2
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
import sys

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#read this path thing https://realpython.com/python-pathlib/
#custom dataset https://www.youtube.com/watch?v=ZoZHd0Zm3RY

plt.ion()   # interactive mode
img_path = os.path.abspath(os.path.join(os.getcwd(), '', 'img'))
spl_path = os.path.abspath(os.path.join(os.getcwd(), '', 'split'))
train_lm = spl_path + '\\train_landmarks.txt'
val_lm = spl_path + '\\val_landmarks.txt'
# landmarks_frame = pd.read_csv(r'C:\Users\SCSE-CIL\PycharmProjects\fashion_attribute\inputs\split\train_landmarks.txt')

# n = 5000
# img_name = str(n)
# landmarks_frame = get_y(data_type = 'train', suffix='_landmarks')
# landmarks = landmarks_frame.iloc[n, :]
# landmarks = np.asarray(landmarks)
# landmarks = landmarks.astype('float').reshape(-1, 2)
#
# print('Image name: {}'.format(img_name))
# print('Landmarks shape: {}'.format(landmarks.shape))
# print('First 4 Landmarks: {}'.format(landmarks[:4]))


#TODO: check the batchsize

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, train=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if train:
            self.landmarks_frame = pd.read_csv(csv_file)
        else:
            self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


#try 10 pic as a small trial first
# def load_data(batchsize=10):
#     '''
#
#     :param data_dir: data directory path
#     :return: train_loader, val_loader, test_loader in PyTorch Loader format
#     '''
#     # insert at 1, 0 is the script path (or '' in REPL)
#     data_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'inputs\img'))
#
#     sys.path.insert(1, data_path)
#
#     data_dir = '../inputs/img'
#     train_dir = data_path + '\\train'
#     valid_dir = data_path + '\\val'
#     print(train_dir)
#
#     # Define transforms for the training, validation, and testing sets
#     training_transforms = transforms.Compose([transforms.RandomRotation(30),
#                                               transforms.RandomResizedCrop(224),
#                                               transforms.RandomHorizontalFlip(),
#                                               transforms.ToTensor(),
#                                               transforms.Normalize([0.485, 0.456, 0.406],
#                                                                    [0.229, 0.224, 0.225])])
#
#     validation_transforms = transforms.Compose([transforms.Resize(256),
#                                                 transforms.CenterCrop(224),
#                                                 transforms.ToTensor(),
#                                                 transforms.Normalize([0.485, 0.456, 0.406],
#                                                                      [0.229, 0.224, 0.225])])
#
#     #TODO: this is how you should get the mean and sd.
#     # loader = DataLoader(train_set, batch_size=len(train_set), num_workers=1)
#     # data[0].mean(), data[0].std()
#     # data = next(iter(loader))
#     # (tensor(0.2860), tensor(0.3530))
#
#     # TODO: Load the datasets with ImageFolder
#     training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
#     validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
#     # testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transforms)
#
#     # TODO: Using the image datasets and the trainforms, define the dataloaders
#     train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batchsize, shuffle=True, num_workers=1)
#     validate_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batchsize, num_workers=1)
#     # test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=batchsize)
#
#     return train_loader, validate_loader #,test_loader

def get_y(data_type='Train', suffix=''):
    '''

    :param data: a string that is either 'train' or 'val'
    :return: attr: the list of label y
    '''
    spl_path = os.path.abspath(os.path.join(os.getcwd(), '', 'split'))
    if data_type == 'Train':
        trial = spl_path+'\Train'+suffix+'.txt'
        print('trial', trial)
        with open(spl_path+'\Train'+suffix+'.txt', 'r') as f:
            attr = f.read()
    elif data_type == 'val':
        with open(spl_path+'\\val'+suffix+'.txt', 'r') as f:
            attr = f.read()
    return attr

# train_loader, validate_loader= load_data()
# print(train_loader)
# #example of loading image as a list
# from PIL import Image
# validation_img_paths = ["validation/alien/11.jpg",
#                         "validation/alien/22.jpg",
#                         "validation/predator/33.jpg"]
# img_list = [Image.open(input_path + img_path) for img_path in validation_img_paths]
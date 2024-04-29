from __future__ import print_function, division
import os
import torch
import torchvision
import pandas as pd
from skimage import io,transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,utils

import warnings
warnings.filterwarnings("ignore")

plt.ion() #interactive mode


# #read annotation CSV
landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')
#
n = 65
img_name = landmarks_frame.iloc[n,0]
landmarks = landmarks_frame.iloc[n,1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1,2)
#
# print('Image name: {}'.format(img_name))
# print('Landmarks shape: {}'.format(landmarks.shape))
# print('First 4 landmarks: {}'.format(landmarks[:4]))
#
#helper function to show the landmarks
def show_landmarks(image,landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:,0],landmarks[:,1],s = 10, marker='.', c='r')
    plt.pause(1)

# plt.figure()
# show_landmarks(io.imread(os.path.join('data/faces/',img_name)),landmarks)
# plt.show()
#--------------------------------------------------------------------------------------

#create a dataset reading the csv
#the dataset sample is a dict


class FaceLandmarkDataset(Dataset):
    def __init__(self,csv_file,root_dir,transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,self.landmarks_frame.iloc[idx,0])
        print(img_name)
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx,1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1,2)
        sample = {'image':image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample



face_dataset = FaceLandmarkDataset(csv_file='data/faces/face_landmarks.csv',
                                   root_dir='data/faces/')

fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i,sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1,4,i+1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break


# transforms
#scale the image
class Rescale(object):
    """Rescale the image in a sample to a given size. """
    def __init__(self,output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image,(new_h,new_w))

        landmarks = landmarks * [new_h/h,new_w/w]
        return {'image':img, 'landmarks': landmarks}

#to crop from image randomly. This is data augmentation
class RandomCrop(object):
    """Crop Randomly the image in a sample"""

    def __init__(self,output_size):
        assert isinstance(output_size, (int,tuple))
        if isinstance(output_size,int):
            self.output_size = (output_size,output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0,h - new_h)
        left = np.random.randint(0,w - new_w)

        image = image[top:top + new_h,
                left:left + new_w]

        landmarks = landmarks - [left,top]

        return {'image': image, 'landmarks': landmarks}

#to convert the numpy images to torch images (we need to swap axes).
class ToTensor(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        #it is needed to swap the color axis because:
        #numpy image = HxWxC
        #tensor image = CxHxW
        image = image.transpose([2,0,1])
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

scale = Rescale(256)
crop = RandomCrop(128)

composed = transforms.Compose([Rescale(256),RandomCrop(224)])

fig = plt.figure()
sample = face_dataset[65]

for i, tsfrm in enumerate([scale,crop,composed]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1,3,i+1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.show()

transformed_dataset = FaceLandmarkDataset(csv_file='data/faces/face_landmarks.csv', root_dir='data/faces',
                                          transform=transforms.Compose([Rescale(256), RandomCrop(224),ToTensor()]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(),sample['landmarks'].size())

    if i==3:
        break


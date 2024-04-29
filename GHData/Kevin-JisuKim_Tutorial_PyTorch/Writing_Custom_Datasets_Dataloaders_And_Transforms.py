from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warning messages
import warnings
warnings.filterwarnings("ignore")

plt.ion() # Interactive mode

# Load csv file and get the annotations in (N, 2) array
landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:]
landmarks = np.asarray(landmarks)
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name : {}'.format(img_name))
print('Landmarks shape : {}'.format(landmarks.shape))
print('First 4 Landmarks : {}'.format(landmarks[:4]))

# Helper function shows an image and its landmarks
def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s = 10, marker = '.', c = 'r')
    plt.pause(1)

plt.figure()
show_landmarks(io.imread(os.path.join('data/faces/', img_name)), landmarks)
plt.show()

# Dataset class (need to override __len__ and __getitem__)
class FaceLandmarksDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform = None):
        self.landmarks_frame = pd.read_csv(csv_file) # Path to the csv file with annotations
        self.root_dir = root_dir # Directory with all the images
        self.transform = transform # Optional transform to be applied on a sample
    
    def __len__(self): # Return a size of dataset
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx): # Find [i]th sample
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image' : image, 'landmarks' : landmarks}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

# Instantiate class and iterate samples
face_dataset = FaceLandmarksDataset(csv_file = 'data/faces/face_landmarks.csv', root_dir = 'data/faces/')

fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()

        break
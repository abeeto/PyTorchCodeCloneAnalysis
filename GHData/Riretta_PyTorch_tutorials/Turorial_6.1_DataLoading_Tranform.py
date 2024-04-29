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

transformed_dataset = FaceLandmarkDataset(csv_file='data/faces/face_landmarks.csv',
                                          root_dir='data/faces/',
                                          transform=transforms.Compose([
                                              Rescale(256),
                                              RandomCrop(224),
                                              ToTensor()
                                          ]))

# for i in range(len(transformed_dataset)):
#     sample = transformed_dataset[i]
#
#     print(i,sample['image'].size(), sample['landmarks'].size())
#
#     if i==3:
#         break

dataloader = DataLoader(transformed_dataset,batch_size=4,shuffle=True,num_workers=4)

def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""

    images_batch, landmarks_batch = sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1,2,0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i,:,0].numpy()+i*im_size,
                    landmarks_batch[i,:,1].numpy(),
                    s=10, marker='.', c='r')

        plt.title('batch from dataloader')

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch,sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break


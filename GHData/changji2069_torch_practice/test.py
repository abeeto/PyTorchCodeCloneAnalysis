from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


# reads the landmarks of a random face
landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')
# n = random.randint(0,65)
n = 54
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1, 2)

# print('Image name: {}'.format(img_name))
# print('Landmarks shape: {}'.format(landmarks.shape))
# print('First 4 Landmarks: {}'.format(landmarks[:4]))

# testing for random crop
# image = io.imread(os.path.join('data/faces/', img_name))
# h, w = image.shape[:2]
# new_h, new_w = 130, 130
#
# top = np.random.randint(0, h-new_h)
# left = np.random.randint(0, w-new_w)
#
# image = image[top:top + new_h, left:left + new_w]
# print(len(landmarks))
# landmarks2 = [[x-left,y-top] for [x,y] in landmarks if left<x<left+new_w and top<y<top+new_h]
# landmarks2 = np.array(landmarks2)
# print(landmarks2)
# print(len(landmarks2))

def show_landmarks(image, landmarks):
    if len(landmarks>0):
        plt.imshow(image)
        plt.scatter(landmarks[:,0], landmarks[:,1], s=10, marker='.', c='r')
        plt.pause(0.001) # pause for update
    else:
        print("Cropped region has no landmark.")
        plt.imshow(image)
        plt.pause(0.001)
# plt.figure()
# show_landmarks(image, landmarks2)
# plt.show()

# creating a custom Dataset class
class FaceLandmarksDataset(Dataset): #inherit Dataset from torch.utils.data.Dataset and override __len__ and __getitem__ methods
    def __init__(self, csv_file, root_dir, transform=None):
        """
        :param csv_file: path to csv file
        :param root_dir: directory with images
        :param transform: optional transform on samples
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx,0])
        image = io.imread(img_name)

        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1,2)

        sample = {'image' : image, 'landmarks' : landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv', root_dir='data/faces/')
#
# fig = plt.figure()
#
# for i in range(len(face_dataset)):
#     sample = face_dataset[i]
#
#     print(i, sample['image'].shape, sample['landmarks'].shape)
#
#     ax=plt.subplot(1,4,i+1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     show_landmarks(**sample) #** is used to unpack the keywords regardless of how many pairs there are
#
#     if i==3:
#         plt.show()
#         break

# preprocessing to resize images to same size

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h>w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        #swapping h and w because x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w/w, new_h/h]

        return {'image':img, 'landmarks':landmarks}

class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)

        image = image[top:top + new_h, left:left + new_w]

        landmarks = np.array([[x-left, y-top] if left<x<left+new_w and top<y<top+new_h else [0,0] for [x,y] in landmarks])

        return {'image':image, 'landmarks':landmarks}

class ToTensor(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # note that
        # numpy image : H x W x C
        # torch image : C x H x W
        image = image.transpose((2,0,1))
        return {'image':torch.from_numpy(image), 'landmarks':torch.from_numpy(landmarks)}

# scale = Rescale(256)
# crop = RandomCrop(70)
# composed = transforms.Compose([Rescale(256), RandomCrop(224)])
#
# fig = plt.figure()
# sample = face_dataset[np.random.randint(0,65)]
#
# for i, tsfrm in enumerate([scale, crop, composed]):
#     transformed_sample = tsfrm(sample)
#
#     ax = plt.subplot(1,3, i+1)
#     plt.tight_layout()
#     ax.set_title(type(tsfrm).__name__)
#     show_landmarks(**transformed_sample) #simply unpacking the dict
#
# plt.show()
transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                           root_dir='data/faces/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)


# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
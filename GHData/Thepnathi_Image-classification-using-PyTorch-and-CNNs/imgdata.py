'''
Assignment 2

author  : Shan Luo
created : 20/11/20 5:30 PM


Edited 01/01/2021 by
    Thepnathi Chindalaksanaloet, 201123978
    Robert Szafarczyk, 201307211
Changes:
    - use opencv instead of scimage to load images,
    - change image labels from 1D numpy arrays to scalar integers,
    - add OpenCV image matrix to the data loader.
'''

import os
import numpy as np
from torch.utils.data import Dataset
import cv2



class imageDataset(Dataset):

    def __init__(self, root_dir, file_path, imSize = 250, shuffle=False):
        self.imPath = np.load(file_path)
        self.root_dir = root_dir
        self.imSize = imSize
        self.file_path=file_path


    def __len__(self):
        return len(self.imPath)

    def __getitem__(self, idx):
        im = cv2.imread(os.path.join(self.root_dir, self.imPath[idx]))  # read the image

        if len(im.shape) < 3: # if there is grey scale image, expand to r,g,b 3 channels
            im = np.expand_dims(im, axis=-1)
            im = np.repeat(im,3,axis = 2)

        img_folder = self.imPath[idx].split('/')[-2]
        if img_folder =='faces':
            label = 0
        elif img_folder == 'dog':
            label = 1
        elif img_folder == 'airplanes':
            label = 2
        elif img_folder == 'keyboard':
            label = 3
        elif img_folder == 'cars':
            label = 4

        img = np.zeros([3,im.shape[0],im.shape[1]]) # reshape the image from HxWx3 to 3xHxW
        img[0,:,:] = im[:,:,0]
        img[1,:,:] = im[:,:,1]
        img[2,:,:] = im[:,:,2]

        imNorm = np.zeros([3,im.shape[0],im.shape[1]]) # normalize the image
        imNorm[0, :, :] = (img[0,:,:] - np.max(img[0,:,:]))/(np.max(img[0,:,:])-np.min(img[0,:,:])) -0.5
        imNorm[1, :, :] = (img[1,:,:] - np.max(img[1,:,:]))/(np.max(img[1,:,:])-np.min(img[1,:,:])) -0.5
        imNorm[2, :, :] = (img[2,:,:] - np.max(img[2,:,:]))/(np.max(img[2,:,:])-np.min(img[2,:,:])) -0.5

        return{
            'im': im,                               # OpenCV image
            'imNorm': imNorm.astype(np.float32),    # normalised image for training
            'label': label                          # image label
            }

class DefaultTrainSet(imageDataset):
    def __init__(self, **kwargs):
        script_folder = os.path.dirname(os.path.abspath(__file__))
         #  img_list_train.npy that contains the path of the training images is provided
        default_path = os.path.join(script_folder, 'img_list_train.npy')
        root_dir = os.path.join(script_folder, 'data')
        super(DefaultTrainSet, self).__init__(root_dir, file_path=default_path, imSize = 250,**kwargs)


class DefaultTestSet(imageDataset):

    def __init__(self, **kwargs):
        script_folder = os.path.dirname(os.path.abspath(__file__))
        #  img_list_test.npy that contains the path of the testing images is provided
        default_path = os.path.join(script_folder, 'img_list_test.npy')
        root_dir = os.path.join(script_folder, 'data')
        super(DefaultTestSet, self).__init__(root_dir, file_path=default_path, imSize = 250,**kwargs)



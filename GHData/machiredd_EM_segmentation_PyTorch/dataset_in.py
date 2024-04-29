import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
import cv2
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class EM_dataset(Dataset):
    def __init__(self, base_dir, in_dir_name, out_dir_name):
        # self.transform = transform  # using transform in torch!
        self.data_dir = base_dir
        self.output_size = (512, 512)
        image_dir = os.path.join(base_dir, in_dir_name, '1')
        target_dir = os.path.join(base_dir, out_dir_name, '1')
        self.image_paths = sorted([join(image_dir, f) for f in listdir(image_dir) if f.endswith('.tif')])
        self.target_paths = sorted([join(target_dir, f) for f in listdir(target_dir) if f.endswith('.png')])
        # self.image_paths = sorted([join(image_dir, f) for f in listdir(image_dir) if isfile(join(image_dir, f))])
        # self.target_paths = sorted([join(target_dir, f) for f in listdir(target_dir) if isfile(join(target_dir, f))])

    def __len__(self):
        return len(self.image_paths)

    def transform(self,image,label):
        w, h = image.size
        pad_w = 2048 - w if 2048 > w else 0
        pad_h = 2048 - h if 2048 > h else 0
        if pad_w != 0 or pad_h != 0:
            image = TF.pad(image, padding=(pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2),
                           padding_mode="reflect")
            label = TF.pad(label, padding=(pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2),
                           padding_mode="reflect")
        # image = image * (1. / 255)
        found = False
        while not found:
            # Random crop
            # [H,W] -> [dim_crop,dim_crop]
            i, j, h, w = transforms.RandomCrop.get_params(image,
                                                          output_size=(2048, 2048))

            img = TF.crop(image, i, j, h, w)
            msk = TF.crop(label, i, j, h, w)
            # print(msk.dtype)
            # image_sub = torch.from_numpy(msk.astype(np.float32)).unsqueeze(0)
            # a = torch.count_nonzero(image_sub)
            # if a>300:
            if msk.getextrema()[1] > 0:
                found = True
                image = img
                label = msk

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)

        # elif random.random() > 0.5:
        #     image, label = random_rotate(image, label)
        image = zoom(image, (self.output_size[0] / 2048, self.output_size[1] / 2048), order=3)  # why not 3?
        label = zoom(label, (self.output_size[0] / 2048, self.output_size[1] / 2048), order=0)
        label = np.where(label > 0.5, 1, 0)
        image = image / 255.
        image = np.expand_dims(image, axis=0)
        # image = np.tile(image,(3,1,1))

        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32)).unsqueeze(0)
 
        return image, label

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = Image.open(self.target_paths[idx])
        image1,label1 = self.transform(image,label)
        sample = [image1,label1]
        return sample
        
        
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample[0], sample[1]

        w, h = image.size
        pad_w = 2048 - w if 2048 > w else 0
        pad_h = 2048 - h if 2048 > h else 0
        if pad_w != 0 or pad_h != 0:
            image = TF.pad(image, padding=(pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2),
                           padding_mode="reflect")
            label = TF.pad(label, padding=(pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2),
                          padding_mode="reflect")
        # image = image * (1. / 255)
        found = False
        while not found:
            # Random crop
            # [H,W] -> [dim_crop,dim_crop]
            i, j, h, w = transforms.RandomCrop.get_params(image,
                                                          output_size=(2048,2048))

            img = TF.crop(image, i, j, h, w)
            msk = TF.crop(label, i, j, h, w)
            # print(msk.dtype)
            # image_sub = torch.from_numpy(msk.astype(np.float32)).unsqueeze(0)
            # a = torch.count_nonzero(image_sub)
            # if a>300:
            if msk.getextrema()[1] > 0:
                found = True
                image = img
                label = msk

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)

        # elif random.random() > 0.5:
        #     image, label = random_rotate(image, label)
        image = zoom(image, (self.output_size[0] / 2048, self.output_size[1] / 2048), order=3)  # why not 3?
        label = zoom(label, (self.output_size[0] / 2048, self.output_size[1] / 2048), order=0)
        label = np.where(label > 0.5, 1, 0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32)).unsqueeze(0)
        image = TF.normalize(image, mean=0, std=255)

        # sample = {'image': image, 'label': label.long()}
        sample = [image,label]
        return sample


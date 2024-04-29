''' Self make OMG Dataset class'''
from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data
import cv2
import torchvision.transforms as transforms
import random

class OMG(data.Dataset):
    """`OMG Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, path,split='Training'):

        self.img_size = 132
        self.crop_size = 128
        self.transform = {
                        'Training':transforms.Compose([
                         transforms.Resize(self.img_size),
                         transforms.RandomCrop(self.crop_size),
                         transforms.RandomHorizontalFlip(),
                         transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0, hue=0.02),
                         transforms.ToTensor(), # 转为Tensor 归一化至0～1
                         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化
                             ]),
                        'Testing': transforms.Compose([
                            # transforms.Resize(self.img_size),
                            # transforms.CenterCrop(self.crop_size),
                            transforms.ToTensor(),  # 转为Tensor 归一化至0～1
                            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化
                        ]),
                            }
        self.split = split  # training set or test set
        self.data = h5py.File(path, 'r', driver='core')
        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = self.data['Training_pixel']
            self.train_labels = self.data['Training_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((34300, self.crop_size, self.crop_size,3))

        else:
            self.test_data = self.data['Testing_pixel']
            self.test_labels = self.data['Testing_label']
            self.test_data = np.asarray(self.test_data)
            self.test_data = self.test_data.reshape((3500, self.crop_size, self.crop_size,3))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
            img = Image.fromarray(img)
            img = self.transform['Training'](img)
        else:
            img, target = self.test_data[index], self.test_labels[index]
            img = Image.fromarray(img)
            img = self.transform['Testing'](img)

        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        else:
            return len(self.test_data)

if __name__ == '__main__':
    train_data=OMG(path='./data/OMG_train_data.h5',split='Training')

    train_loader = data.DataLoader(dataset=train_data,
                                   batch_size=32,
                                   shuffle=True,
                                   num_workers=0)

    print(len(train_data))
    # for i,(img,label) in enumerate(train_data):
    #     if i<1:
    #         img=np.transpose(np.array(img),(1,2,0))
    #         print(img.shape)
    #         img=(img*0.5+0.5)*255
    #         cv2.imwrite('1.jpg',img)
    #         print(label.shape)
    for i,(img, label) in enumerate(train_loader):
        print(img.shape)
        print(label.view(-1,1))
        if i<1:
            print('train')
            img=np.transpose(np.array(img)[0],(1,2,0))
            img = (img) * 255
            cv2.imwrite('34.jpg',img)
            print(label)
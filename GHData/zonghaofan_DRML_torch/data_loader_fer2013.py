''' Fer2013 Dataset class'''
from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data
import cv2
import torchvision.transforms as transforms
import random
# # 定义对数据的预处理
# transform = transforms.Compose([
#         transforms.ToTensor(), # 转为Tensor 归一化至0～1
#         #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化
#                              ])
class FER2013(data.Dataset):
    """`FER2013 Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, path,split='Training'):
        self.img_size = 50
        self.crop_size = 48
        self.transform = {'Training': transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.RandomCrop(self.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # 转为Tensor 归一化至0～1
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化
        ]),
            'Testing': transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),  # 转为Tensor 归一化至0～1
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化
            ])}
        self.split = split  # training set or test set
        self.data = h5py.File(path, 'r', driver='core')
        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = self.data['Training_pixel']
            self.train_labels = self.data['Training_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((29779, 48, 48))

        # elif self.split == 'PublicTest':
        #     self.PublicTest_data = self.data['PublicTest_pixel']
        #     self.PublicTest_labels = self.data['PublicTest_label']
        #     self.PublicTest_data = np.asarray(self.PublicTest_data)
        #     self.PublicTest_data = self.PublicTest_data.reshape((3000, 48, 48))

        else:
            self.PrivateTest_data = self.data['Testing_pixel']
            self.PrivateTest_labels = self.data['Testing_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((3323, 48, 48))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
        # elif self.split == 'PublicTest':
        #     img, target = self.PublicTest_data[index], self.PublicTest_labels[index]
        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img[:, :, np.newaxis]
        # img=np.concatenate((img,img,img),axis=-1)

        #detail augmentation
        s = np.concatenate((img, img, img), axis=-1).astype(np.float32)

        b1 = cv2.GaussianBlur(s, (3, 3), 0)

        D1 = (s - b1) + s
        D1 = cv2.resize(D1, (self.crop_size, self.crop_size))
        img=np.clip(D1, 0, 255).astype(np.uint8)

        img = Image.fromarray(img)

        if self.split == 'Training':
            img = self.transform['Training'](img)
        else:
            img = self.transform['Testing'](img)
        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        # elif self.split == 'PublicTest':
        #     return len(self.PublicTest_data)
        else:
            return len(self.PrivateTest_data)

if __name__ == '__main__':
    train_data=FER2013(path='./data/fer2013/data/fer2013_new_data.h5',split='Training')

    train_loader = data.DataLoader(dataset=train_data,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=0)

    print(len(train_data))

    #test
    test_data = FER2013(path='./data/fer2013/data/fer2013_new_data.h5', split='Testing')

    test_loader = data.DataLoader(dataset=test_data,
                                   batch_size=8,
                                   shuffle=True,
                                   num_workers=0)

    print(len(test_data))

    # for i,(img,label) in enumerate(train_data):
    #     if i<1:
    #         img=np.transpose(np.array(img),(1,2,0))
    #         print(img.shape)
    #         img=(img*0.5+0.5)*255
    #         cv2.imwrite('1.jpg',img)
    #         print(label.shape)

    for i,(img, label) in enumerate(train_loader):
        print(img.shape)
        print(img.dtype)
        print(label.view(-1,1))
        if i<2:
            print('train')
            img=np.transpose(np.array(img)[0],(1,2,0))
            img = (img) * 255
            cv2.imwrite('3.jpg',img)
            print(label)

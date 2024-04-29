import os
import glob
import random

import numpy as np

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


def make_datapath_list(phase='train'):
    root_path = './data'
    target_path = os.path.join(root_path, phase, '*/*.jpg')
    path_list = []
    label_list = []
    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list


def make_datapath_dic(phase='train'):
    root_path = './data/' + phase
    class_list = os.listdir(root_path)
    class_list = [class_name for class_name in class_list if not class_name.startswith('.')]
    datapath_dic = {}
    for i, class_name in enumerate(class_list):
        data_list = []
        target_path = os.path.join(root_path, class_name, '*.jpg')
        for path in glob.glob(target_path):
            data_list.append(path)
        datapath_dic[i] = data_list

    return datapath_dic


class ImageTransform():
    def __init__(self, resize):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


class TripletDataset(Dataset):
    def __init__(self, datapath_dic, transform=None, phase='train'):
        self.datapath_dic = datapath_dic
        self.transform = transform
        self.phase = phase

        all_datapath = []
        bins = [0]
        for data_list in self.datapath_dic.values():
            all_datapath += data_list
            bins.append(bins[-1] + len(data_list))
        self.all_datapath = all_datapath
        self.bins = bins

    def __len__(self):
        return len(self.all_datapath)

    def __getitem__(self, idx):
        anchor_path = self.all_datapath[idx]
        for i in range(len(self.bins)):
            if idx < self.bins[i]:
                positive_pathlist = self.all_datapath[self.bins[i-1]:self.bins[i]]
                negative_pathlist = self.all_datapath[:self.bins[i-1]] + self.all_datapath[self.bins[i]:]
                anchor_label = i
                break

        positive_path = random.choice(positive_pathlist)
        negative_path = random.choice(negative_pathlist)

        anchor = self.transform(Image.open(anchor_path), self.phase)
        positive = self.transform(Image.open(positive_path), self.phase)
        negative = self.transform(Image.open(negative_path), self.phase)

        return anchor, positive, negative, anchor_label


class MyDataset(Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img, self.phase)
        if 'NAYEON' in img_path:
            label = 0
        elif 'JEONGYEON' in img_path:
            label = 1
        elif 'MOMO' in img_path:
            label = 2
        elif 'SANA' in img_path:
            label = 3
        elif 'JIHYO' in img_path:
            label = 4
        elif 'MINA' in img_path:
            label = 5
        elif 'DAHYUN' in img_path:
            label = 6
        elif 'CHAEYOUNG' in img_path:
            label = 7
        elif 'TZUYU' in img_path:
            label = 8

        return img_transformed, label

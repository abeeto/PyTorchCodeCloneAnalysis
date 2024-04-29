import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as Transforms
import glob
import cv2


class Mydatasets(Dataset):
    def __init__(self, train=True, transform=None):
        super(Mydatasets, self).__init__()
        # self.train = train
        if train:
            # img_dir = r'E:\data\MNIST\img\train_img'
            label_txt = '/home/chauncy/data/MNIST/img/train_label.txt'
        else:
            # img_dir = r'E:\data\MNIST\img\test_img'
            label_txt = '/home/chauncy/data/MNIST/img/test_label.txt'

        labels = open(label_txt)
        self.images = []
        for line in labels:
            line = line.strip().split()
            self.images.append((line[0], int(line[1])))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = cv2.imread(self.images[item][0])
        label = self.images[item][1]
        if self.transform:
            img = self.transform(img)
        return img, label





import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tfs
import torchvision.transforms.functional as TF
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader

root = "PATH_TO_DATA_AND_LABEL"

def read_images(root, train=True):
    txt_fname = ("./train.txt" if train else "./val.txt")
    with open(txt_fname, "r") as f:
        images = f.read().split()
    data = [os.path.join(root, "img", i+".jpg") for i in images]
    label = [os.path.join(root, "label", i+".png") for i in images]
    return data, label

def BGR2RGB(img):
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    return img

def image2label(im):
    classes = ['background','object']
    colormap = [[0,0,0],[52,0,255]]
    cm2lbl = np.zeros(256**3)
    for i,cm in enumerate(colormap):
        cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i
    data = np.array(im, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64')

def img_transforms(im, label, inference=False):
    im_tfs = tfs.Compose([tfs.ToTensor(), tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    im = im_tfs(im)
    if not inference:
        label = image2label(label)
        label = torch.from_numpy(label)
        label = nn.ZeroPad2d(padding=(6,6,6,6))(label)
        label = torch.nn.functional.one_hot(label, 2).permute(2,0,1)
    return im, label

class SegDataset(Dataset):
    def __init__(self, train, transforms):
        self.transforms = transforms
        self.data_list, self.label_list = read_images(root=root, train=train)

    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = cv2.imread(img)
        img = BGR2RGB(img)
        label = cv2.imread(label)
        label = BGR2RGB(label)
        img, label = self.transforms(img, label)
        return img, label, self.data_list[idx]

    def __len__(self):
        return len(self.data_list)      

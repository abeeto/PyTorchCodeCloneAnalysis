# ///////////////////////////////////////////////////////////////
# BY: FURKAN EREN
# 01/11/2021
# V: 1.0.0
# ///////////////////////////////////////////////////////////////
import cv2
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class ImageDataset(Dataset):

    def __init__(self, files_raw,files_label):
        self.files_raw = files_raw
        self.files_label = files_label

        mean = np.array([0.485])
        std = np.array([0.229])

        #Transforms for low resolution images and high resolution images

        self.raw_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.label_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        temp_label = Image.open(self.files_label[index % len(self.files_label)]).convert('L')
        temp_raw = Image.open(self.files_raw[index % len(self.files_raw)]).convert('L')
        img_raw = self.raw_transform(temp_raw)
        img_label = self.label_transform(temp_label)
        return img_raw,img_label

    def __len__(self):
        return len(self.files_label)
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import os

def load_image(path):
    return Image.open(path)

# This dataset use the image datasets where images
# are located on the dataset folder and this
# Generic dataset should be customized according to the dataset
# csv based dataset require different loading function

class GenericDataset(Dataset):
    """ Initialize the dataset by giving the dataset path and transform that will be applied """
    def __init__(self,data_path = '.',transform = None):
        images = []
        subjects = [subject for subject in os.listdir(data_path)]
        for subject in subjects:
            subject_path = os.path.join(data_path,subject)
            for x in os.listdir(subject_path):
                images.append((os.path.join(subject_path, x), int(subject) - 1))

        self.images = images
        self.transform = transform
        self.count = len(images)
        self.class_count = len(subjects)

    """ Image with given index will be loaded by using the image path """
    def __getitem__(self, index):
        image_path,label = self.images[index]
        image = load_image(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return image,label

    def __len__(self):
        return self.count

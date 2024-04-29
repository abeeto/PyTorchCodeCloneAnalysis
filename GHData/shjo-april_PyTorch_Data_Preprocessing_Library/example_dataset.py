import cv2
import numpy as np

from torchvision import transforms
from data.reader import SH_Dataset

root_dir = 'F:/Classification_DB_Sang/'
dataset_name = 'Cars'

train_transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(0.5),
])

train_dataset = SH_Dataset(root_dir + f'{dataset_name}/train/*.sang', train_transform)

for image, label in train_dataset:
    image = np.asarray(image)[..., ::-1]
    print(image.shape, label)
    
    cv2.imshow('show', image)
    cv2.waitKey(0)

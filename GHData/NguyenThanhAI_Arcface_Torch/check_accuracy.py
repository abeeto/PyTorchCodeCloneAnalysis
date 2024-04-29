import argparse
import os

import math

import random

from itertools import groupby

from PIL import Image
from requests import get

from tqdm import tqdm

import numpy as np

import torch
from torch import optim
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

from backbones import get_model

from utils_fn import enumerate_images

import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = r"D:\Face_Datasets\CelebA_Models\checkpoint.pth"
#images_dir = r"D:\Face_Datasets\choose_train_torchattacks_images\AutoAttack"
images_dir = r"D:\Face_Datasets\choose_train"



class FaceModel(nn.Module):

    def __init__(self, model_name: str="r18", num_classes: int=10177):
        super().__init__()
        self.backbone = get_model(name=model_name)

        for layer in self.backbone.parameters():
            layer.requires_grad = False
        

        #in_features = self.backbone.features.out_features
        self.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, images):
        x = self.backbone(images)
        output = self.fc(x)

        return output


class FaceDataset(Dataset):
    def __init__(self, images_dir: str, subset: str="train") -> None:
        super().__init__()
        self.images_list = enumerate_images(images_dir=images_dir)
        self.class_list = list(set(list(map(lambda x: os.path.normpath(x).split(os.sep)[-2], self.images_list))))
        self.class_list.sort()
        self.class_to_label = dict(zip(self.class_list, range(len(self.class_list))))
        print(self.class_to_label)
        #print(self.class_to_label)
        self.transfrom = transforms.Compose([transforms.Resize([112, 112]), transforms.ToTensor()])
    
    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image = self.images_list[index]
        label = int(self.class_to_label[os.path.normpath(image).split(os.sep)[-2]])

        #img = torchvision.io.read_image(image)
        img = Image.open(image).convert("RGB")
        
        #img.div_(255).sub_(0.5).div_(0.5)
        img = self.transfrom(img)

        return img, label


def val_loop(dataloader: DataLoader, model: nn.Module, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batch = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            pred = model(images)

            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
        
    test_loss /= num_batch
    correct /= size
    print("Accuracy: {}%, Avg loss: {}".format(correct * 100, test_loss))
    return correct, test_loss

dataset = FaceDataset(images_dir=images_dir)

dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=False)

model = FaceModel(model_name="r50", num_classes=len(dataset.class_list))

model.load_state_dict(torch.load(weights, map_location=torch.device("cpu"))["weights"])
model.to(device=device)

model.eval()

loss_fn = nn.CrossEntropyLoss()

val_loop(dataloader=dataloader, model=model, loss_fn=loss_fn)
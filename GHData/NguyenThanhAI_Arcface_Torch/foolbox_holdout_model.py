import os

import argparse

from typing import List
from tqdm import tqdm

import numpy as np

import cv2

import torch
import torch.optim as optim

import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torch.nn as nn


import foolbox

from backbones import get_model
from utils_fn import enumerate_images, load_images_and_labels_into_tensors


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FaceModel(nn.Module):

    def __init__(self, model_name: str="r18", num_classes: int=10177):
        super().__init__()
        self.backbone = get_model(name=model_name)

        #for layer in self.backbone.parameters():
        #    layer.requires_grad = False
        

        #in_features = self.backbone.features.out_features
        self.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, images):
        x = self.backbone(images)
        output = self.fc(x)

        return output


weights = r"D:\Face_Datasets\CelebA_Models\checkpoint.pth"
images_dir = r"D:\Face_Datasets\choose_train"
save_dir = r"D:\Face_Datasets\choose_train_foolbox_images"

model = FaceModel(model_name="r18", num_classes=len(identities))
model.load_state_dict(torch.load(weights, map_location=torch.device("cpu"))["weights"])
model.eval()
model.to(device=device)
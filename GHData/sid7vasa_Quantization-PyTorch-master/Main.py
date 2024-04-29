import torch
from torch import nn
from torchvision import models
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

from imutils import paths
import cv2
import os
import numpy as np
from tqdm import tqdm
import datetime
import random
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import copy
from utils.train import train_model

device = ("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda":
    print("-"*40, "We are powered by GPU", "-"*40)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class CDDataset(Dataset):
    def __init__(self, df, transform=None):
        self.image_paths = df['images']
        self.labels = df['labels']
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(
            "data/dogs-vs-cats/train/train/", self.image_paths[idx]))
        img = img.resize((128, 128))
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


image_paths = list(paths.list_images("./data/dogs-vs-cats/train/train"))
print(len(image_paths))
random.shuffle(image_paths)
names = []
labels = []
for i in tqdm(image_paths):
    name = Path(i).name
    label = name.split(".")[0]
    if label == "cat":
        label = 0
    if label == "dog":
        label = 1
    labels.append(label)
    names.append(name)
train = pd.DataFrame({"images": pd.Series(
    names[:23000]), "labels": pd.Series(labels[:23000])})
val = pd.DataFrame({"images": pd.Series(
    names[23000:]), "labels": pd.Series(labels[23000:])})
del names, image_paths, labels

image_datasets = {}
batch_size = 4
image_datasets['train'] = CDDataset(train, data_transforms['train'])
image_datasets['val'] = CDDataset(val, data_transforms['val'])
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                              shuffle=True)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = ["cat", "dog"]
print(dataset_sizes, class_names)


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()


# Get a batch of training data
inputs, classes = next(iter(dataloaders['val']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


model_ft = torchvision.models.resnet18(pretrained=False)
for param in model_ft.parameters():
    param.requires_grad = True
num_ftrs = model_ft.fc.in_features
model_ft.fc = torch.nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)
criterion = torch.nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.001)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.001)
pytorch_total_params = sum(p.numel() for p in model_ft.parameters())
print("Total Model Parameters:", pytorch_total_params)
pytorch_trainable_params = sum(p.numel()
                               for p in model_ft.parameters() if p.requires_grad)
print("Total Trainable Parameters", pytorch_trainable_params)

model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=2)

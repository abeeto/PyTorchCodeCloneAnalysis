# -*- coding: utf-8 -*-
"""
====================================================================================================================
Title: train.py
Description: Training a neural network model over the captured images. When the code is executed, the model is built
over the captured training images, and evaluated over the validation set images.
====================================================================================================================
"""

__author__ = "Aparajit Balaji"

# Importing necessary modules

import os
import time
from PIL.Image import Image

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchinfo import summary

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(123)

# Setting the suitable transformations to be applied on the images

img_transforms = {
    "train": transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])}

# Setting the path parameters and initializing the datasets

set_dirs = {"train": os.path.join("images", "train"), "val": os.path.join("images", "val")}

image_datasets = {x: datasets.ImageFolder(set_dirs[x], transform = img_transforms[x]) for x in set_dirs.keys()}
dataloaders = {x: DataLoader(image_datasets[x], batch_size = 50, shuffle = True) for x in set_dirs.keys()}
sizes = {x: len(image_datasets[x]) for x in set_dirs.keys()}

# Test code - please choose to skip this section

# ====================================================================

# for x in set_dirs.keys():
#     for idx, data in enumerate(image_datasets[x]):
#         try:
#             os.mkdir("transformed_images\{}\{}".format(x, data[1]))
#         except FileExistsError:
#             # print("File already exists!")
#             pass
#         Image.save(transforms.ToPILImage()(data[0].squeeze_(0)), 
# 'transformed_images\{}\{}\{}.png'.format(x, data[1], idx))

# ====================================================================

# Printing the gestures along with their indices

gestures = {class_name: class_index for class_index, class_name in image_datasets["train"].class_to_idx.items()}
print(gestures)

# Initializing a pretrained model, customizing the final classifier layer, and freezing all but the final layer  

model = models.alexnet(pretrained = True)

model.classifier = nn.Sequential(nn.Linear(in_features = 9216, out_features = 2048),
                                  nn.Linear(in_features = 2048, out_features = 512),
                                  nn.Linear(in_features = 512, out_features = 5))

for param in model.parameters():
    param.requires_grad = False

for params in model.classifier.parameters():
    params.requires_grad = True

# Printing the model configuration

print("\n", model)
print("\n", summary(model, (50, 3, 224, 224)))

# Setting the criterion function and the optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# Training and validation processes start

epochs = 2
start = time.time()
best_acc = 0.0

for epoch in range(epochs):

    print(f'Epoch #{epoch + 1} of {epochs}, Learning Rate = {[group["lr"] for group in optimizer.param_groups]}')
    print('-----' * 10)

    for phase in ['train', 'val']:

        if phase == 'train':
            model.train()
        else:
            model.eval()
                                                
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in dataloaders[phase]:
                        
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, prediction = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(prediction == labels.data)
        
        epoch_loss = running_loss / sizes[phase]
        epoch_acc = (running_corrects.item() / sizes[phase]) * 100
        print(f'{phase} loss: {epoch_loss : .6f} {phase} accuracy = {epoch_acc : .3f}%')
        
            
    print('-----' * 10)
       
    if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc

print(f'Training complete in {(time.time() - start) // 60 : .0f} mins {(time.time() - start) % 60 : .0f} seconds')
print(f'Best Validation Accuracy: {best_acc : .3f}%')

# Saving the model trained

torch.save(model, "gesture_model.pth")
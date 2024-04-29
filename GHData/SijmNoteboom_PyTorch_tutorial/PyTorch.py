import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from sklearn.metrics import confusion_matrix
# from plotcm import plot_confusion_matrix

# import pdb

torch.set_printoptions(linewidth=120)


train_set = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1000, shuffle=True)

print(len(train_set))
print(len(train_loader))

sample = next(iter(train_set))
image, label = sample

plt.imshow(image.squeeze(), cmap="gray")
plt.show()
torch.tensor(label)

train_set.train_labels.bincount()
train_set.targets.bincount()


display_loader = torch.utils.data.DataLoader(train_set, batch_size=10)
batch = next(iter(display_loader))
print('len:', len(batch))

images, labels = batch

print('types:', type(images), type(labels))
print('shapes:', images.shape, labels.shape)

grid = torchvision.utils.make_grid(images, nrow=10)

plt.figure(figsize=(15, 15))
plt.imshow(np.transpose(grid, (1, 2, 0)))
plt.show()

print('labels:', labels)

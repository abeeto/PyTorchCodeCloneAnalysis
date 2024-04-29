import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

"""
PyTorch has two primitives to work with data: torch.utils.data.DataLoader and torch.utils.data.Dataset. 
Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset.
"""

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size=64
train_dataset = DataLoader(training_data, batch_size=batch_size)
test_dataset = DataLoader(test_data, batch_size=batch_size)

for x, y in test_dataset:
    print(f"train shape: {x.shape}, datatype: {x.dtype}")
    print(f"test label: {y.shape} datatype: {y.dtype}")
    break
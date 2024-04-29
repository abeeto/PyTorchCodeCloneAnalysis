import torch
import labels
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

"""
@description: Download data sets from web and store in Tensor
"""
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
"""
@description: Randomly samples / visualizes training data
"""


def visualize_sample():
    labels_map = labels.fashion_map
    figure = plt.figure(figsize=(8, 8))

    cols, rows = 5, 5
    dim = rows * cols
    for i in range(1, dim + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]

        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


"""
@description: Creates dataloader and generates random data batch
"""


def gen_batch():
    batch_size = 64

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return [train_dataloader, test_dataloader]

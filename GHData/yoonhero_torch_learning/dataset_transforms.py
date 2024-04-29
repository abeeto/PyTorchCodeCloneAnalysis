import torch
import torchvision

dataset = torchvision.datasets.MNIST(
    root="./data", transform=torchvision.transforms.ToTensor()
)


# practice for working with datasets and dataloaders in PyTorch
import torch
from torch.utlis.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# define training data, aquired from torchvision datasets
training_data = datasets.FashionMNIST(
    root = "data", train = True,
    download = True, transform = ToTensor()
)

# define testing data
test_data =  datasets.FashionMNIST(
    root = "data", train=False,
    download =True, transform = ToTensor()
)

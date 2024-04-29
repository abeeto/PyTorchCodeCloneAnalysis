from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.utils.data as data


class Dataset:
    def __init__(self, manual_seed: int, device: str):
        self.use_cuda = device == "cuda"
        torch.manual_seed(manual_seed)

        self.trans = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.2435, 0.2616])
        ])

    def get_train_valid_loaders(self, train_ratio: float, batch_size: int):
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.trans)
        data_len = len(train_set)
        train_size = int(data_len * train_ratio)
        valid_size = data_len - train_size

        train_set, valid_set = data.random_split(train_set, [train_size, valid_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=self.use_cuda)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, pin_memory=self.use_cuda)
        return train_loader, valid_loader

    def get_test_loader(self, batch_size: int):
        test_set = datasets.CIFAR10(root='./data', download=True, train=False, transform=self.trans)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=self.use_cuda)
        return test_loader

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.utils.data as data


class Dataset:
    def __init__(self, manual_seed: int, device: str):
        self.use_cuda = device == "cuda"
        torch.manual_seed(manual_seed)

        self.trans = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def get_train_valid_loaders(self, train_ratio: float, batch_size: int):
        train_set = datasets.MNIST(root='./MNIST', download=True, train=True, transform=self.trans)

        data_len = len(train_set)
        train_size = int(data_len * train_ratio)
        valid_size = data_len - train_size

        train_set, valid_set = data.random_split(train_set, [train_size, valid_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=self.use_cuda)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, pin_memory=self.use_cuda)
        return train_loader, valid_loader

    def get_test_loader(self, batch_size: int):
        test_set = datasets.MNIST(root='./MNIST', download=True, train=False, transform=self.trans)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=self.use_cuda)
        return test_loader

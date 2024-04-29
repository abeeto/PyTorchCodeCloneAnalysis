from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize


def get_data_loaders(train_batch_size, val_batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(MNIST(download=True,
                                    root=".",
                                    transform=data_transform,
                                    train=True),
                              batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(MNIST(download=False,
                                  root=".",
                                  transform=data_transform,
                                  train=False),
                            batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader

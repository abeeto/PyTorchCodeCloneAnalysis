from torch.utils.data import DataLoader

from torchvision.datasets.mnist import MNIST 
from torchvision.transforms import ToTensor


def load_dataset(batch_size, eval = False):
    transform = ToTensor()
    if not eval:
        train_set = MNIST(root='./datasets/', train=True, download=True, transform=transform)
        train_load = DataLoader(train_set, shuffle=True, batch_size=batch_size)   
        test_set = MNIST(root='./datasets/', train=False, download=True, transform=transform)
        test_load = DataLoader(test_set, shuffle=False, batch_size=batch_size)
        return train_load, test_load

    else:
        test_set = MNIST(root='./datasets/', train=False, download=True, transform=transform)
        test_load = DataLoader(test_set, shuffle=True, batch_size=1)
        return test_load
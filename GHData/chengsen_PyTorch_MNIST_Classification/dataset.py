from torch.utils.data import DataLoader
from torchvision import datasets, transforms

__build__ = 2018
__author__ = "singsam_jam@126.com"


def get_loader(args, kwargs):
    train_loader = DataLoader(dataset=
                              datasets.MNIST('./data', train=True, download=True,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))
                                             ])),
                              batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(dataset=
                             datasets.MNIST('./data', train=False, transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ])),
                             batch_size=args.test_batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader

import os
import torchvision

if not os.path.exists('./data'):  #
    os.mkdir('./data')

data_train = torchvision.datasets.MNIST(
    root='./data',
    download=True,
    train=True,
    transform=torchvision.transforms.ToTensor(),
)

data_valid = torchvision.datasets.MNIST(
    root='./data',
    download=True,
    train=False,
    transform=torchvision.transforms.ToTensor(),
)



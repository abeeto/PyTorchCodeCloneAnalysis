import torch 
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from torchvision import datasets, transforms
import torch.utils.data as data

import numpy as np
from PIL import Image
import os
import os.path

class SubsetSampler(Sampler):
    
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.arange(len(self.indices)))

    def __len__(self):
        return len(self.indices)

def get_loader_svhn(num_train, batch_size, train):
    """Builds and returns Dataloader for SVHN dataset."""
    transform = transforms.Compose([
                    transforms.Scale((32,32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if train:
        split = 'train'
    else:
        split = 'test'
    svhn = datasets.SVHN(root='./datasets/svhn', split=split, download=True, transform=transform)
    
    indices = list(range(num_train))
    np.random.shuffle(indices)
    sampler = SubsetSampler(indices)

    svhn_loader = torch.utils.data.DataLoader(dataset=svhn,
                                              batch_size=batch_size,
                                              sampler=sampler,
                                              num_workers=1,
                                              pin_memory=True)
    return svhn_loader

def get_loader_mnist(num_train, batch_size, train):
    """Builds and returns Dataloader for MNIST dataset."""
    transform = transforms.Compose([
                    transforms.Scale((32,32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    mnist = datasets.MNIST(root='./datasets/mnist', train=train, download=True, transform=transform)
    
    indices = list(range(num_train))
    np.random.shuffle(indices)
    sampler = SubsetSampler(indices)

    mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                              batch_size=batch_size,
                                              sampler=sampler,
                                              num_workers=1,
                                              pin_memory=True)
    return mnist_loader


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):
    
    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)

def get_loader(config, train=True):
    num_train   = config['num_train']
    batch_size  = config['batch_size']
    img_size    = config['img_size']
    num_workers = config['num_workers']

    if train:
        dataset_path = config['src_dataset_train']
    else:
        dataset_path = config['src_dataset_test']

    if dataset_path == 'mnist':
        return get_loader_mnist(num_train, batch_size, train)
    elif dataset_path == 'svhn':
        return get_loader_svhn(num_train, batch_size, train)
    
    transform_list = [
                    transforms.Scale(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5))
    ]
    transform = transforms.Compose(transform_list)

    dataset = ImageFolder(dataset_path, transform=transform)

    indices = list(range(num_train))
    np.random.shuffle(indices)
    sampler = SubsetSampler(indices)

    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    return train_loader

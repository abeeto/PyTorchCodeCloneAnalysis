import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mnist_dataloader(config, train=True):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    dataset = datasets.MNIST(
        config.data_path,
        train=train,
        download=True,
        transform=transform,
    )

    return torch.utils.data.DataLoader(dataset,
                                       batch_size=config.batch_size,
                                       shuffle=True)


def get_optimizer(optimizer):
    optimizer_dict = {
        'adam': torch.optim.Adam,
    }
    
    return optimizer_dict[optimizer]


def get_criterion(criterion):
    criterion_dict = {
        'crossentropy': torch.nn.CrossEntropyLoss(),
    }

    return criterion_dict[criterion]


class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

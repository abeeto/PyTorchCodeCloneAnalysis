import numpy as np
import torch
from torchvision import datasets, transforms


def load_split_train_test(data_directory, valid_size=.2):
    train_transforms = transforms.Compose([transforms.Resize([224,224]),
                                           transforms.ToTensor(),
                                           ])
    test_transforms = transforms.Compose([transforms.Resize([224, 224]),
                                          transforms.ToTensor(),
                                          ])
    train_data = datasets.ImageFolder(data_directory,
                                      transform=train_transforms)
    test_data = datasets.ImageFolder(data_directory,
                                     transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                                              sampler=train_sampler, batch_size=1)
    # Using a batch_size = 1, each image is its own tensor
    testloader = torch.utils.data.DataLoader(test_data,
                                             sampler=test_sampler, batch_size=1)
    return trainloader, testloader

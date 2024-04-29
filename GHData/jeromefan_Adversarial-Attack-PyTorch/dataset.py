import json
from torchvision import datasets, transforms
import torch.utils.data


def cifar10():
    mean = (0.491, 0.482, 0.447)
    std = (0.202, 0.199, 0.201)

    val_data = datasets.CIFAR10("data/cifar10", train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)
                                ]))
    tensor_mean = torch.tensor(mean).view(3, 1, 1)
    tensor_std = torch.tensor(std).view(3, 1, 1)
    return val_data, tensor_mean, tensor_std


def setupDataset(chosenDataset):

    with open('config.json') as config_file:
        config = json.load(config_file)
    batch_size = int(config['batch_size'])

    if chosenDataset == 'cifar10':
        val_data, mean, std = cifar10()
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
    else:
        raise Exception("其他数据集暂不支持! 请使用README中所述的已支持数据集！")

    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, drop_last=True)
    return val_dataloader, classes, mean, std

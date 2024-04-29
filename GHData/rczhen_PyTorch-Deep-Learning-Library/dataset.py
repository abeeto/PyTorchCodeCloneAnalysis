from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import datasets


def get_transforms(mode='train'):
    if mode == 'train':
        data_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])

    return data_transforms


def get_dataset(name='cifar10', train=True):
    if name == 'cifar10':
        dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=get_transforms())

    return dataset


def get_dataloader(dataset, batch_size=128, mode='train'):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=(mode == 'train'))

    return dataloader


if __name__ == "__main__":
    """
    Test dataset functions. Not for actual training. 
    """

    dataset = get_dataset(train=True)
    dataloader = get_dataloader(dataset, batch_size=16, mode='train')
    for batch_id, data in enumerate(dataloader):
        image = data[0]
        label = data[1]
        print(type(data))
        print(type(label), label.shape)
        print(type(image), image.shape)
        acc = (label == label).float().sum() / label.shape[0]
        print(acc)
        break

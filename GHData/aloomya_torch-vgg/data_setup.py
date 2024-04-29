import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def data_loader(
    data_dir,
    batch_size,
	input_width=224,
    valid_size=0.1,
    shuffle=True,
    test=False):

    transform = transforms.Compose([
		transforms.Resize((input_width,input_width)),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])

    if test:
        dataset = datasets.CIFAR100(
            root=data_dir, train=False,
            download=True, transform=transform,
        )

        data_loader = data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

        return data_loader

    dataset = datasets.CIFAR100(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    num_train = len(dataset)
    split = int(valid_size * num_train)
    
    train_dataset, valid_dataset = data.random_split(dataset,
                                                     (num_train - split, split))

    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle)

    valid_loader = data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=shuffle)

    return (train_loader, valid_loader)

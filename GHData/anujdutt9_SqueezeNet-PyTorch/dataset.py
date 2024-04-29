from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt


def get_data_loader(BATCH_SIZE, image_transforms, num_workers=2):
    # Dataset
    train_dataset = CIFAR10(root='./', train=True, download=True, transform=image_transforms)
    val_dataset = CIFAR10(root='./', train=False, download=True, transform=image_transforms)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)

    return train_loader, val_loader


if __name__ == '__main__':
    # Dataset Transforms
    transform = transforms.Compose(
        [transforms.Resize(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5],
                              [0.5, 0.5, 0.5]),
         ])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Test DataLoaders
    train_loader, val_loader = get_data_loader(BATCH_SIZE=32,
                                               image_transforms=transform,
                                               num_workers=2)

    # TEST DataLoader
    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    img = img.permute(1, 2, 0)
    img = img.numpy()
    img = img.clip(min=0, max=1)
    plt.title(classes[label])
    plt.imshow(img)
    plt.show()
    print(f"Label: {classes[label]}")

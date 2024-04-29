import torchvision, os
from transforms import get_transforms

def get_dataset(IMG_SIZE = 224):

    train_root = os.path.join(os.getcwd(), '../CamSDD/training')
    val_root = os.path.join(os.getcwd(), '../CamSDD/validation')
    test_root = os.path.join(os.getcwd(), '../CamSDD/test')

    train_transforms, val_transforms, test_transforms = get_transforms(IMG_SIZE)

    train_dataset = torchvision.datasets.ImageFolder(root=train_root, transform=train_transforms)
    val_dataset = torchvision.datasets.ImageFolder(root=val_root, transform=val_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root=test_root, transform=test_transforms)

    return (train_dataset, val_dataset, test_dataset)

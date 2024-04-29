from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def get_train_transformations() -> transforms:
    """Get transformations for training images."""
    return transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def get_validation_transformations() -> transforms:
    """Get transformations for training images."""
    return transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def load_data(train_data_path: str, validation_data_path: str, batch_size: int) -> (DataLoader, DataLoader, int):
    """Load train and validation data, return it's loaders and number of total training batches.

    :param train_data_path path to images used for train
    :param validation_data_path path to images used for validation
    :param batch_size number of images in single batch
    """
    train_dataset = datasets.ImageFolder(train_data_path, transform=get_train_transformations())
    validation_dataset = datasets.ImageFolder(validation_data_path, transform=get_validation_transformations())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    total_training_batches = int(len(train_dataset) / batch_size)

    return train_loader, validation_loader, total_training_batches

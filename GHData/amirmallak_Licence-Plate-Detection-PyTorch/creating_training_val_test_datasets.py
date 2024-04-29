import torch
import config
import pandas as pd
import torchvision.transforms as transforms

from loading_image_label_pair import CarLicensePlateDataset
from torch.utils.data import DataLoader, random_split


def _create_datasets(dataset: pd.DataFrame, width: int, height: int) -> tuple:
    # Building the Training, Validation, and Test datasets
    data_dir = config.data_dir

    mean = [0.485, 0.456, 0.406]  # Mean of ImageNet dataset
    std = [0.229, 0.224, 0.225]  # STD of ImageNet dataset

    normalize = transforms.Normalize(mean=torch.Tensor(mean), std=torch.Tensor(std))
    interpolation_mode = transforms.InterpolationMode.BICUBIC
    transform: transforms = transforms.Compose([
        transforms.Resize((width, height), interpolation=interpolation_mode),
        transforms.ToTensor(),
        normalize])

    # Load Data
    tensor_dataset = CarLicensePlateDataset(data_dir, dataset, transform)
    tensor_dataset_len = len(tensor_dataset)  # tensor_dataset.__len__()
    val_size = int(tensor_dataset_len * 0.1)
    train_size = tensor_dataset_len - val_size
    test_size = int(val_size - (val_size // 2))  # 5% of the data
    train_dataset, val_dataset = random_split(tensor_dataset, [train_size, val_size])
    val_dataset, test_dataset = random_split(val_dataset, [val_size // 2, test_size])

    batch_size = int(config.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=None)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, sampler=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, sampler=None)
    test_batch_loader = DataLoader(test_dataset, batch_size=12, shuffle=False, sampler=None)

    return train_loader, val_loader, test_loader, test_batch_loader, train_dataset, val_dataset, test_dataset

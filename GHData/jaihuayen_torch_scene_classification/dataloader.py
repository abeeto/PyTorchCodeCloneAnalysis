from torch.utils.data import DataLoader
import torchvision
import os
from dataset import get_dataset

def get_loader(BATCH_SIZE = 64, IMG_SIZE = 224):

    train_dataset, val_dataset, test_dataset = get_dataset(IMG_SIZE)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return (train_dataloader, val_dataloader, test_dataloader)
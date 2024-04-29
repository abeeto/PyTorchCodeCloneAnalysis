import os
import glob
from PIL import Image

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

def load_data(file_dir, batch_size, key):

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((1, 1, 1), (1, 1, 1))
    ])

    files_cat1 = glob.glob(os.path.join(file_dir, '*' + key[0] + '*'))
    files_cat2 = glob.glob(os.path.join(file_dir, '*' + key[1] + '*'))

    X = []
    Y = []
    for file in files_cat1:
        X_img = Image.open(file)
        X.append(transform(X_img))
    for file in files_cat2:
        Y_img = Image.open(file)
        Y.append(transform(Y_img))

    dataset = TensorDataset(torch.stack(X), torch.stack(Y))
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader

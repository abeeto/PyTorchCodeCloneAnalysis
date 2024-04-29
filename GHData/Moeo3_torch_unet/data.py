from PIL import Image
import os
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader


def read_train_data(data_path = './train', feature = 'image', label = 'label'):
    feature_path = os.path.join(data_path, feature)
    label_path = os.path.join(data_path, label)

    features = []
    for png in os.listdir(feature_path):
        img = Image.open(os.path.join(feature_path, png))
        img = np.asarray(img)[None, :, :]
        features.append(img / 255)
        pass
    features = np.asarray(features)
    
    labels = []
    for png in os.listdir(label_path):
        img = Image.open(os.path.join(label_path, png))
        img = np.asarray(img)[None, :, :]
        labels.append(img // 255)
        pass
    labels = np.asarray(labels)

    return features, labels

def train_data_load(batch_size = 3):
    x, y = read_train_data()
    train_dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return train_data_loader

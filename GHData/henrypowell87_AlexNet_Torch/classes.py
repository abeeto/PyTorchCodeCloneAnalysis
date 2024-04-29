import torch
import torch.nn as nn
from torch.utils import data
from PIL import Image


# Dataset: dataset generator. Loads batches of dataset on CPU and then feeds them to GPU during training.
class DataSet(data.Dataset):
    def __init__(self, list_IDs, labels, data_dir, transform=None):
        # Initialization function
        self.labels = labels
        self.list_IDs = list_IDs
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        # Gets total number of samples
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Generates on sample of data
        ID = self.list_IDs[index]

        x = Image.open(tensor_path + ID)
        if self.transform:
            x = self.transform(x)
        y = self.labels[ID]

        return x, y


# AelxNet Model definition
class Net(nn.Module):
    def __init__(self, num_classes=103):
        super(Net, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=256 * 6 * 6,  out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

# Define forward pass
    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


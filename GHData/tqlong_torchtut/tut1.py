import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from easydict import EasyDict
import numpy as np
from tqdm import tqdm

from trainer import train_model

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(7*7*32, num_classes)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
def get_datasets():
    datasets = EasyDict(dict(
        train=torchvision.datasets.MNIST(root='data/', train=True, download=True, transform=transforms.ToTensor()),
        test=torchvision.datasets.MNIST(root='data/', train=False, download=True, transform=transforms.ToTensor()),
    ))
    return datasets

def main():
    train_config = EasyDict(dict(
        # device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        # training configuration
        num_epochs = 5,
        num_classes = 10,
        batch_size = 100,
        learning_rate = 1e-3,
    ))

    datasets = get_datasets()
    model = ConvNet(num_classes=train_config.num_classes).to(train_config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    train_model(train_config, model, optimizer, criterion, datasets)
    torch.save({'model': model.state_dict()}, 'runs/convnet_exp_1.ckpt')
    
if __name__ == "__main__":
    main()

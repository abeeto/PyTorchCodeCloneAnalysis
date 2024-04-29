from pickletools import optimize
import numpy as np
import gc
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.tensorboard import SummaryWriter

def data_loader(
    data_dir,
    batch_size,
    random_seed=321,
    valid_size=0.1,
    shuffle=True,
    test=False
):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize,
    ])

    if test:
        dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

        return data_loader

    # Load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform
    )

    num_train = int(len(train_dataset) / 5)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler
    )

    return (train_loader, valid_loader)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer_0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer_1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer_2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer_3 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv_1(x)
        # x = self.maxpool(x)
        x = self.layer_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

num_classes = 10
num_epochs = 20
batch_size = 256
learning_rate = 0.01
writer = SummaryWriter(f"runs/lr{learning_rate}_bs{batch_size}")

# CIFAR10 dataset
train_loader, valid_loader = data_loader(data_dir="/home/visha/cifar10", batch_size=batch_size)
test_loader = data_loader(data_dir="/home/visha/cifar10", batch_size=batch_size, test=True)

for data, label in train_loader:
    print(data.shape)
    break

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001, momentum=0.9)

# Training
total_step = len(train_loader)
total_valid_steps = len(valid_loader)

for epoch in range(num_epochs):
    with tqdm(total=total_step) as pbar:
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the confirued device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i == 0:
                if epoch == 0:
                    writer.add_graph(model, images)

                writer.add_scalar(
                    "train/loss",
                    loss,
                    epoch
                )

            # Cleaning memory
            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()

            pbar.set_postfix_str(f'loss: {loss.item()}')
            pbar.update(1)

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        with tqdm(total=total_valid_steps) as pbar_valid:
            for valid_step, (images, labels) in enumerate(valid_loader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if valid_step == total_valid_steps - 1:
                    writer.add_scalar(
                        "valid/accuracy",
                        100 * correct/total,
                        epoch
                    )

                del images, labels, outputs

                pbar_valid.update(1)
        
        print(f'Accuracy of the network on the {total_valid_steps} validation images: {100 * correct/total} %')
        
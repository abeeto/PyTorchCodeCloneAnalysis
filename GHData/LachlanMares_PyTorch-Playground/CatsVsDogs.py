import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

load_dataset = False

IMAGE_SIZE = 50
IMAGE_CHANNELS = 3
CATEGORIES = 2
BATCH_SIZE = 128
EPOCHS = 0

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


class CovNet(nn.Module):
    def __init__(self, image_channels=IMAGE_CHANNELS, image_size=IMAGE_SIZE, kernel_size=3, pool_size=(2,2) ,categories=CATEGORIES):
        super().__init__()
        self.categories = categories
        self.image_channels = image_channels
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.image_height = image_size
        self.image_width = image_size

        self.conv1 = nn.Conv2d(self.image_channels, 32, self.kernel_size)
        self.conv2 = nn.Conv2d(32, 64, self.kernel_size)
        self.conv3 = nn.Conv2d(64, 128, self.kernel_size)

        x = torch.randn(self.image_height * self.image_width * self.image_channels).view(-1, self.image_channels, self.image_height, self.image_width)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, self.categories)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), self.pool_size)
        x = F.max_pool2d(F.relu(self.conv2(x)), self.pool_size)
        x = F.max_pool2d(F.relu(self.conv3(x)), self.pool_size)

        if self._to_linear is None:
            #self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
            self._to_linear = 128 * 26 * 26
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=0)
        return x


class CatsVsDogs():
    def __init__(self, image_size=50, batch_size=BATCH_SIZE):
        self.image_size = image_size
        self.data_path = "Cat_Dog_data"
        self.train_loader = []
        self.test_loader = []
        self.batch_size = batch_size

        train_transforms = transforms.Compose([transforms.Resize(self.image_size),
                                               transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor()])

        test_transforms = transforms.Compose([transforms.Resize(self.image_size),
                                              transforms.ToTensor()])

        train_data = datasets.ImageFolder(self.data_path + "/train", transform=train_transforms)
        test_data = datasets.ImageFolder(self.data_path + "/test", transform=test_transforms)

        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=True)


cats_v_dogs = CatsVsDogs(image_size=IMAGE_SIZE)


covnet = CovNet().to(device)

print(covnet)

optimiser = optim.Adam(covnet.parameters(), lr=0.003)
loss_function = nn.NLLLoss()

for epoch in range(EPOCHS):
    for inputs, labels in cats_v_dogs.train_loader:

        # Shift computation to GPU
        inputs, labels = inputs.to(device), labels.to(device)

        # Training pass
        optimiser.zero_grad()
        outputs = covnet(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimiser.step()

    print("loss", loss)


with torch.no_grad():
    test_loss = 0
    accuracy = 0
    for inputs, labels in cats_v_dogs.test_loader:
        # Shift computation to GPU
        inputs, labels = inputs.to(device), labels.to(device)
        output = covnet(inputs)
        loss = loss_function(output, labels)
        test_loss += loss.item()

        # Calculate accuracy
        output = torch.exp(output)
        top_ps, top_class = output.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print("Test loss: {:.8f}..".format(test_loss / len(cats_v_dogs.test_loader)),
          "Test Accuracy: {:.8f}".format(accuracy / len(cats_v_dogs.test_loader)))

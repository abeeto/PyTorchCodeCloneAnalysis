import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

# Step1: load dataset
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# Step2: make dataset iterable
batch_size = 100
n_iters = 300
n_epochs = n_iters / (len(train_dataset / batch_size))
n_epochs = int(n_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Step3: create model class
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Conv1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        # Average pool
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)

        # Conv2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        # Average pool
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)

        # Fully connected 1 (read out)
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, *input):
        # con1
        output = self.cnn1(input)
        output = self.relu1(output)
        #         avg pool
        output = self.avgpool1(output)

        # con1
        output = self.cnn2(output)
        output = self.relu2(output)
        #         avg pool
        output = self.avgpool2(output)

        output = output.view(output.size(0), -1)

        output = self.fc1(output)
        return output


# Step4: instantiate model class
model = CNNModel()
# Step5: instantiate loss class
criterion = nn.CrossEntropyLoss()
# Step6: instantiate optimizer class
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# Parameters in depth

# Step7: train model
iter = 0
for epoch in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #         Load image variable
        images = Variable(images)
        labels = Variable(labels)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        iter += 1

        if iter % 500 == 0:
            correct = 0
            total = 0

            for images, labels in test_loader:
                images = Variable(images)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            accuracy = 100 * correct / total
            print('{} {} {}'.format(iter, loss.data[0], accuracy))
            

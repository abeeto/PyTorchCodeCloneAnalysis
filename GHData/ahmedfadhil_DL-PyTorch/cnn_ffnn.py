import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

# Step1: load dataset
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

print(train_dataset.train_data.size())
print(train_dataset.train_labels.size())
print(test_dataset.test_data.size())
print(test_dataset.test_labels.size())

# Step2: make dataset iterable
batch_size = 100
n_iter = 3000
num_epochs = n_iter / (len(train_dataset / batch_size))
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Step3: create model class
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        #         Fully connected 1 (readout)
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

        def forward(self, x):
            # Convolution 1
            out = self.cnn1(x)
            out = self.relu1(out)

            #             Max pool 1
            out = self.maxpool1(out)

            # Convolution 2
            out = self.cnn2(out)
            out = self.relu2(out)

            #             Max pool 2
            out = self.maxpool2(out)

            #             Resize
            out = out.view(out.size(0), -1)

            #             Linear function (readout)
            out = self.fc1(out)
            return out


# Step4: instantiate model class
model = CNNModel()
# Step5: instantiate loss class
criterion = nn.CrossEntropyLoss()
# Step6: instantiate optimizer class
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Parameters in depth
print(model.parameters())
print(len(list(model.parameters())))
print(list(model.parameters())[0].size())
print(list(model.parameters())[1].size())
print(list(model.parameters())[2].size())
print(list(model.parameters())[3].size())
print(list(model.parameters())[4].size())
print(list(model.parameters())[5].size())
# Step7: train model
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Load images as Variables
        images = Variable(images)
        labels = Variable(labels)

        # Clear gradients wrt parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)
        # Calculate loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        # Getting gradients wrt parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter +=1

        if iter %500==0:
            correct = 0
            total = 0

            # iterate through dataset
            for images,labels in test_loader:
                # load images to a Torch variable
                images= Variable(images)

                # Forward pass only to get logits/output
                outputs = model(images)

                # Getting predictions from the max values
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total+=labels.size(0)

#                 Total correct predictions
                correct+=(predicted == labels).sum()
            accuracy = 100*correct/total

            print(iter, loss.data[0], accuracy)

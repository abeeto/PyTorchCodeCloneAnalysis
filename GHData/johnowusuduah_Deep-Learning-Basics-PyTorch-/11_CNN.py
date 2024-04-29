# NOTES
# *****************************************************************************
# Applies Convolutional Layers Which Comprise of Convolutional Filters
# and Pooling Layer

# Convolutional Filters ---> Used to Extract Specific Features from Input Data

# Pooling Layers ---> Approach to down sample feature maps by summarizing the
#                     presence of features in patches of the feature map. Also helps
#                     to avoid overfitting by providing an abstracted form of the
#                     input.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np

# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# i. HYPERPARAMETERS
# **********************************************************************************************************************************************
num_epochs = 4
batch_size = 4
learning_rate = 0.001


# ii. DATA
# **********************************************************************************************************************************************
# dataset has PILImage images of range [0, 1].
# We transform them to Tensors of range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ("plane", "car", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")
print(f"No. of iterations in train loader {len(train_loader)}")


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))


# iii. CONVOLUTIONAL NEURAL NETWORK
# **************************************************************************************************************************************************
# MODEL ARCHITECTURE
# __________________Feature Learning_________________________________    ___________________________________Classification___________________________
# Input -> Convolution+ReLU -> Pooling -> Convolution+ReLU -> Pooling -> Flatten -> Fully Connected -> Softmax(Already Included in Crossentropy loss)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # convolutional layer 1 --> input channel size = 3, output channel size = 6 and kernel size = 5)
        self.conv1 = nn.Conv2d(3, 6, 5)
        # pooling layer --> kernel size = 2 and stride = 2
        self.pool = nn.MaxPool2d(2, 2)
        # convolutional layer 2 ---> input channel size must be equal to the last channel output size and output size = 16
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 3 layer fully connected neural net layer
        # no. of channels is increased from 3 to 6 and then to 16
        # 120 and 84 node sizes can be changed but the input node size of 16*5*5 and output node size of 10 must be fixed
        # why 16*5*5
        # use [(W-F+2P)/S) + 1] to compute the size of the resulting images after applying filters (refer to green book)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # original shape [4, 3, 32, 32] = [4, 3, 1024]
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [(i+1)/{n_total_steps}], Loss: {loss.item():0.4f}")


print("Finished Training")


# iv. TESTING AND EVALUATION
# ********************************************************************************************************************************************************
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value, index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]

            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    # compute accuracy
    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy of the network: {acc}%")

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f"Accuracy of {classes[i]}: {acc} %")

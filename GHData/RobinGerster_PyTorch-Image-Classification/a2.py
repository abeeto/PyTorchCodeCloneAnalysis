import glob
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision import datasets, transforms, models
import shutil
import os

train_dir = r'C:\Users\Gerst\PycharmProjects\csc420a2\notMNIST_small\train'
valid_dir = r'C:\Users\Gerst\PycharmProjects\csc420a2\notMNIST_small\validation'
test_dir = r'C:\Users\Gerst\PycharmProjects\csc420a2\notMNIST_small\test'


input_size = 784  # 28x28
decay = 0
num_classes = 10
num_epochs = 4
batch_size = 100
learning_rate = 0.001
####################################


transform = transforms.Compose([
        #gray scale
        transforms.Grayscale(),
        #resize
        transforms.Resize((28,28)),
        #converting to tensor
        transforms.ToTensor(),
        #normalize
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

#neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        #Note a soft max is not nedded because the cross entropy loss already applies this

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

#neural network with two hidden layers
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(hidden_size, num_classes)
        #Note a soft max is not nedded because the cross entropy loss already applies this

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

#neural network with one hidden layer and dropout
class NeuralNet3(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet3, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(loader, model):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        acc = 0
        for images, labels in loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            acc = 100.0 * n_correct / n_samples
        return(acc)

n_total_steps = len(train_loader)

def train(model, criterion, optimizer):
    train_acc = []
    valid_acc = []
    test_acc = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        train_acc.append(accuracy(train_loader, model))
        valid_acc.append(accuracy(valid_loader, model))
        test_acc.append(accuracy(test_loader, model))

    print("Training Acc: " + str(train_acc))
    print("Validation Acc: " + str(valid_acc))
    print("Testing Acc: " + str(test_acc))

    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(valid_acc, label='Validation Accuracy')
    plt.plot(test_acc, label='Testing Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()



def model1():
    model = NeuralNet(input_size, 1000, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    train(model, criterion, optimizer)


def model2():
    model = NeuralNet2(input_size, 500, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    train(model, criterion, optimizer)

def model3():
    model = NeuralNet3(input_size, 1000, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    train(model, criterion, optimizer)

model3()
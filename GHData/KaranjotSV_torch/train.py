import pdb
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils import device, accuracy
from networks import NN, CNN, RNN


arch = "RNN"
# pdb.set_trace()
size = 784 if arch == "NN" else 28  # number of sequences in a sample i.e time steps
seq_len = 28  # number of features in each time step
layers = 2
hidden = 256  # nodes in a hidden

channels = 1
classes = 10
rate = 0.001
batch = 64
epochs = 2

# data
train = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train, batch_size=batch, shuffle=True)

test = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test, batch_size=batch, shuffle=True)

# model = NN(input_size=size, classes=classes).to(device)
# model = CNN().to(device)
model = RNN(size, hidden, layers, seq_len, classes, device).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=rate)

# train network
for epoch in range(epochs):
    for batch_ind, (data, targets) in enumerate(train_loader):
        # data to device
        if arch == "RNN":
            data = data.squeeze(1)  # batchx1x28x28 to batchx28x28
        data = data.to(device)
        targets = targets.to(device)

        if arch == "NN":
            # reshape
            data = data.reshape(data.shape[0], -1)
        # forward
        preds = model(data)
        loss = criterion(preds, targets)
        # backward
        optimizer.zero_grad()  # to set all gradients to 0 for each batch
        loss.backward()
        # gradient descent
        optimizer.step()  # updation of weights based upon gradients calculated in loss.backward()

# check accuracy
accuracy(train_loader, model, arch)
accuracy(test_loader, model, arch)

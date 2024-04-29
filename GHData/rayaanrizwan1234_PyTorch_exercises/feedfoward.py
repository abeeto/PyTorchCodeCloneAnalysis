# MNIST
# DataLoader, Transformation
# Multilayer Neural Net, activation function
# Loss and Optimizer
# training Loop (batch training)
# Model evaluation
# GPU support

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 784  # 28x28
# hidden layer
hidden_size = 500
# 0 to 9 digits
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
# makes it an iterable obj
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
# return an iterable
examples = iter(test_loader)
example_images, example_labels = examples.next()

print(example_images.shape, example_labels.shape)

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(example_images[i][0], cmap='gray')


# plt.show()

class NeuralNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, num_classes):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, num_classes)

    def forward(self, X):
        out = self.linear1(X)
        out = F.relu(out)
        # we dont apply softmax or sigmoid here because cross entropy does it for us
        out = self.linear2(out)
        return out


model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training looop
num_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # images = 100, 1, 28, 28
        # 100, 784 size image. Reshaping to 100 by 784 because its 100 batch size and one image has 784 bits of data cause 28x28
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # forward pass
        y_pred = model(images)

        # loss
        loss = criterion(y_pred, labels)

        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'epoch: {epoch + 1}/{num_epochs}, step {i + 1}/{num_total_steps}, loss  = {loss.item()}')

# test
# the model is already trained with the correct parameters
# so now we test the model with another dataset (test_dataset)
with torch.no_grad():
    for images, labels in test_loader:
        n_correct = 0
        n_samples = 0
        images = images.reshape(-1, 28 * 28).to(device)
        labels.to(device)
        output = model(images)

        # return value, index of the correct label
        _, predictions = torch.max(output, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
        acc = 100 * n_correct / n_samples
    print(f'acc = {acc}')
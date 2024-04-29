import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('/Users/lukasrois/tensorboard')

# hyper parameters

num_epochs = 8
batch_size = 64
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                           transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                          transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                          shuffle=False)

# examples = iter(test_loader)
# images, labels = examples.next()
# print(images.shape, labels.shape)
# torch.Size([64, 1, 28, 28]) torch.Size([64])


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.deep_layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = x.view(x.size(0), -1)
        x = self.deep_layers(x)
        return x


model = NeuralNetwork()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
losses = []
epochs = []
for epoch in range(num_epochs):
    running_loss = 0.0
    running_correct = 0
    for i, (images, labels) in enumerate(train_loader):

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, prediction = torch.max(outputs, 1)
        running_correct += (prediction == labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f"epoch: {epoch + 1}, step: {i + 1}/{n_total_steps}, loss: {running_loss / 64:.4f}")
            print(f"Accuracy: {running_correct / 64}")
            losses.append(running_loss / 64)
            epochs.append(epoch * n_total_steps + i)
            writer.add_scalar('accuracy', running_correct / 64, epoch * n_total_steps + i)
            running_correct = 0
            running_loss = 0.0

plt.plot(epochs, losses)
plt.show()
# test

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, prediction = torch.max(outputs, 1)
        n_correct += (prediction == labels).sum().item()
        n_samples += 64
    acc = 100.0 * n_correct / n_samples
    print(f"Test Accuracy: {acc}")

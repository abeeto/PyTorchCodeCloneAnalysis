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

print(train_dataset.train_data.size())
print(train_dataset.train_labels.size())
print(test_dataset.test_data.size())
print(test_dataset.test_labels.size())

# Make dataset iterable
batch_size = 100
n_iter = 3000
n_epochs = n_iter / (len(train_dataset) / batch_size)
n_epochs = int(n_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Build the mode
class RnnModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RnnModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Num of hidden layers
        self.hidden_layer = layer_dim

        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        h0 = Variable(torch.zeros(self.layer_dim, input.size(0), self.hidden_dim))
        output, hn = self.rnn(input, h0)
        output = self.fc(output[:, -1:])
        return output


#     Instantiate model class
input_dim = 28
hidden_dim = 100
layer_dim = 1  # Change it to two or more for more layers
output_dim = 10

model = RnnModel(input_dim, hidden_dim, layer_dim, output_dim)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# Parameter in-depth
len(list(model.parameters()))

list(model.parameters())[0].size()
list(model.parameters())[2].size()
list(model.parameters())[1].size()
list(model.parameters())[3].size()
list(model.parameters())[4].size()

# Or just do the above with a for loop
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i]).size()

seq_dim = 28
iter = 0
for n_epoch in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #         Load images as variables
        images = Variable(images.view(-1, seq_dim, input_dim))
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
                images = Variable(images.view(-1, seq_dim, input_dim))
                outputs = model(images)
                _, predicted = torch.max(outputs.data)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            accuracy = 100 * correct / total
            print(iter, loss.data[0], accuracy)

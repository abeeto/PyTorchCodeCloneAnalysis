import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plot
import collections

train_dataset = dsets.MNIST(root='./data/MNIST',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=False)

test_dataset = dsets.MNIST(root='./data/MNIST',
                           train=False,
                           transform=transforms.ToTensor())

# Plotting a random image
randInstance = int(np.random.rand(1) * len(train_dataset))
show_img = train_dataset[randInstance][0].numpy().reshape(28,28)
plot.imshow(show_img, cmap = 'gray')
plot.xlabel('Its ' + str(int(train_dataset[randInstance][1])))
plot.show()

batch_size = 100
n_iters = 6000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)
input_dim = 28*28
output_dim = 10

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

isinstance (train_loader,collections.Iterable)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation_layer):
        super(LogisticRegressionModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_layers[0])
        self.linears = nn.ModuleList([nn.Linear(hidden_layers[i-1], hidden_layers[i]) for i in range(1, len(hidden_layers))])
        # Following won't work
        # for i in range(1, len(hidden_layers)):
        #     self.linears.append(nn.Linear(hidden_layers[0], hidden_layers[0]))
        self.output_layer = nn.Linear(hidden_layers[-1], output_dim)
        if activation_layer == 'relu':
            self.activation_layer = nn.ReLU()
        elif  activation_layer == 'sigmoid':
            self.activation_layer = nn.Sigmoid()

    def forward(self,x):
        out = self.input_layer(x)
        out = self.activation_layer(out)
        for i in range(len(self.linears)):
            out = self.linears[i](out)
            out = self.activation_layer(out)
        out = self.output_layer(out)
        return out

hidden_layers = [1500, 2500, 2000]
activation_layer = 'relu'

model = LogisticRegressionModel(input_dim, hidden_layers, output_dim, activation_layer)

if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()

learning_rate= 0.004
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        if torch.cuda.is_available():
            images = Variable(images.view(-1, input_dim).cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images.view(-1, input_dim))
            labels = Variable(labels)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1
        if iter %100 == 0:
            print('Iteration: {}. Loss: {} '.format(iter, loss.data[0]))

        if iter %500 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:

                if torch.cuda.is_available():
                    images = Variable(images.view(-1, input_dim).cuda())
                else:
                    images = Variable(images.view(-1, input_dim))

                # Forward pass only to get logits/output
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                correct += (predicted.cpu() == labels.cpu()).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.data[0], accuracy))

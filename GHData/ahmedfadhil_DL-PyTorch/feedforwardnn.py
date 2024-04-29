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
n_iters = 3000
num_epochs = n_iters / (len(train_dataset / batch_size))
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Step3: create model class
class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNeuralNetwork, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linear function
        self.sigmoid = nn.Sigmoid()
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # Linear function
        out = self.fc1(x)
        # Non-linearity
        out = self.sigmoid(out)
        # Linear function
        out = set.fc2(out)
        return out


# Step4: instantiate model class
input_dim = 28 * 28
hidden_dim = 100
output_dim = 10
model = FeedForwardNeuralNetwork(input_dim, hidden_dim, output_dim)

# Step5: instantiate loss class
criterion = nn.CrossEntropyLoss()

# Step6: instantiate optimizer class
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(model.parameters())
print(list(model.parameters()))
print(len(list(model.parameters())))
# FC1 parameters
print(list(model.parameters())[0].size())

# FC1 Bias parameters
print(list(model.parameters())[1].size())
# FC2 parameters
print(list(model.parameters())[2].size())

# FC1 Bias parameters
print(list(model.parameters())[3].size())

# Step7: train model
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataset):
        # convert inputs/labels to variables
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        # clear gradients buffers
        optimizer.zero_grad()

        # get output given input
        outputs = model(images)

        # get loss
        loss = criterion(outputs, labels)
        # get gradients wrt parameters
        loss.backward()
        # update parameters using gradients
        optimizer.step()
        # repeat for num of epochs
        iter += 1

        if iter % 500 == 0:
            #             Calculate accuracy
            correct = 0
            total = 0
            #             Iterate through test dataset
            for images, labels in test_loader:
                # Load images to torch variable
                images = Variable(images.view(-1, 28 * 28))

                #                 Forward pass to get output
                outputs = model(images)
                #                 Get prediction from max values
                _, predicted = torch.max(outputs.data, 1)
                #                 Total number of labels
                total += labels.size(0)
                #                 Total correct predictions
                correct += (predicted == labels).sum()
            accuracy = 100 * correct / total

            #             Print loss
            print("Iteration: {}, Loss: {}, Accuracy: {}".format(iter, loss.data[0], accuracy))


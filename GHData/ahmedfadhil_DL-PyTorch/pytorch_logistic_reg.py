import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

len(train_dataset)

train_dataset[0]

# Input matrix
train_dataset[0][0].size()

# Label
train_dataset[0][1].size()

# Display dataset
train_dataset[0][0].numpy().shape

show_img = train_dataset[0][0].numpy().reshape(28, 28)
plt.imshow(show_img, cmap='gray')

# Load test dataset
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
len(test_dataset)
type(test_dataset[0])
# Image matrix
test_dataset[0][0].size()
# Label
test_dataset[0][1]

len(train_dataset)

batch_size = 100
n_iters = 3000

n_epochs = n_iters / (len(train_dataset) / batch_size)
n_epochs = int(n_epochs)

# Create iterable object: Training Dataset
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Create iterable object: Testing Dataset
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

import collections

isinstance(train_loader, collections.Iterable)  # True
isinstance(test_loader, collections.Iterable)  # True

# Main aim of iteration
img_1 = np.ones((28, 28))
img_2 = np.ones((28, 28))

lst = [img_1, img_2]

for i in lst:
    print(i.shape)


#     Build a linear regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


#     Instantiate model
input_dim = 28 * 28
output_dim = 10

model = LogisticRegressionModel(input_dim, output_dim)

# Instantiate a loss class
criterion = nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Parameters in depth
print(model.parameters())
print(len(list(model.parameters())))

# FC 1 parameters
print(list(model.parameters())[0].size())

# FC1 Bias parameters
print(list(model.parameters())[1].size())

# Train model
iter = 0

for epoch in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Load images as variables
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        # Clear gradients wrt to params
        optimizer.zero_grad()

        #         Forward pass to get output/logits
        outputs = model(images)
        # Calculate loss: softmax - cross entropy
        loss = criterion(outputs, labels)
        # Getting gradients wrt parameters
        loss.bacward()

        # Update parameters
        optimizer.step()

        iter += 1
    if iter % 500 == 0:
        # Calculate accuracy
        correct = 0
        total = 0

        # Iterate through test dataset
        for images, labels in test_loader:
            # Load images to a torch variable
            images = Variable(images.view(-1, 28 * 28))
            #             forward pass only to get logits/outputs
            outputs = model(images)
            #             Get predictions from the max value
            _, predictions = torch.max(outputs.data, 1)

            # Total number of labels
            total += labels.size(0)

            # Total correct predictions
            correct += (predictions == labels).sum()

        accuracy = 100 * correct / total

        # print loss
        print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.data[0], accuracy))

iter_test = 0
for images, labels in test_loader:
    iter_test += 1
    images = Variable(images.view(-1, 28, 28))
    outputs = model(images)
    if iter_test == 1:
        print('output')
        print(outputs)
        # Output size
        print(outputs.size())
        # First image output for all 10 possibilities
        print(outputs[0, :])
        # The output prediction == hint: 7
        print(predictions[0, :])
        # The label prediction == hint: 7
        print(labels[0])
    _, predicted = torch.max(outputs.data, 1)

# How does the .sum() function work

a = np.ones((10))
print(a)
b = np.ones((10))
print(b)

print(a == b)
print((a == b).sum())

# Save model
save_model = False
if save_model is True:
    #     Saves only parameters
    torch.save(model.state_dict(), 'model.pkl')

#     For a GPU version
if torch.cuda.is_available():
    model.cuda()

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #         GPU or CPU
        if torch.cuda.is_available():
            # model.cuda()
            images = Variable(images.view(-1, 28 * 28).cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images.view(-1, 28 * 28), 1)
            labels = Variable(labels)

        #             total correct predictions
        if torch.cuda.is_available():
            correct += (predictions.cpu() == labels.cpu()).sum()

        else:
            correct += (predictions == labels).sum()
    accuracy = 100 * correct / total

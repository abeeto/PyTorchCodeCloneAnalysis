import helper
import torch
from torch import nn
from torchvision import datasets, transforms
from torch import optim
import numpy as np
import matplotlib.pyplot as plt


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to the first hidden layer linear transformation
        self.hidden_1 = nn.Linear(784, 128)
        # Inputs to the second hidden layer linear transformation
        self.hidden_2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(64, 10)

        # Define sigmoid activation function, softmax, and cross-entropy loss
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden_1(x)
        x = self.relu(x)
        x = self.hidden_2(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.logsoftmax(x)

        return x

 # Get the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5,), std=(0.5,), inplace=False)])
trainset = datasets.MNIST('MNIST_data/', download=True,
                          train=True, transform=transform)
testset = datasets.MNIST('MNIST_data/', download=True,
                         train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)

# Define and run network
model = Network()
# Define loss function
criterion = nn.NLLLoss()
# Optimizer require parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a vector with a length of 784
        images = images.view(images.shape[0], -1)
        # Clear the gradients, do this becuase gradients are accumulated
        optimizer.zero_grad()
        # Forward pass
        output = model.forward(images)
        # Calculate loss
        loss = criterion(output, labels)
        # Backward pass
        loss.backward()
        # Update the weights
        optimizer.step()
        # Keep track of loss
        running_loss += loss.item()
    else:
        print('Training loss: {}'.format(running_loss/len(trainloader)))


images, labels = next(iter(trainloader))

img = images[0].view(1, 784)

# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model.forward(img)

# Output of the network are logits, need to take softmax for probabilities
ps = torch.exp(logps)
helper.view_classify(img.view(1, 28, 28), ps)

########################################################################################################################
########################################################################################################################
# TESTING #
correct_count, all_count = 0, 0
for images, labels in testloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model.forward(img)

        ps = torch.exp(logps)
        probs = list(ps.numpy()[0])
        pred_label = probs.index(max(probs))
        true_label = labels.numpy()[i]
        if true_label == pred_label:
            correct_count += 1
        all_count += 1

print("Number of images tested: \n", all_count)
print("Model accuracy: ", correct_count/all_count)

########################################################################################################################
########################################################################################################################
# Independent matrix multiplication
W1 = model.hidden_1.weight.detach().numpy()
B1 = model.hidden_1.bias.detach().numpy()
W2 = model.hidden_2.weight.detach().numpy()
B2 = model.hidden_2.bias.detach().numpy()
WO = model.output.weight.detach().numpy()
BO = model.output.bias.detach().numpy()


def relu(x):
    """ Numpy implementation of ReLU"""
    # 1st method:
    return np.maximum(x, 0)
    # 2nd method (faster)
    # return x * (x > 0)


def log_softmax(x):
    """Softmax function

    Arguments
    ---------
    x: numpy array
    """
    den = np.sum(np.exp(x))
    return np.log(np.exp(x)/den)


images, labels = next(iter(trainloader))
img = images[0].view(1, 784)

# Model outoput
with torch.no_grad():
    logps = model.forward(img)

# Manual multiplication
img = img.numpy()
layer1 = np.dot(img, W1.T) + B1
layer1 = relu(layer1)
layer2 = np.dot(layer1, W2.T) + B2
layer2 = relu(layer2)
output = np.dot(layer2, WO.T) + BO
output = log_softmax(output)

plt.plot(logps.T, label='model')
plt.plot(output.T, label='manual')
plt.legend()
plt.show()

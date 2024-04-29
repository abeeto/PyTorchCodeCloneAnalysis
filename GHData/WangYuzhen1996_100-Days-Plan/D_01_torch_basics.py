#--------------------------------------------------#
# Some part of the code referenced from below      #
# https://github.com/yunjey/pytorch-tutorial       #
#--------------------------------------------------#
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

#===========================================#
#                 Sample 1                  #
#              Basic autograd 1             #
#                  python3                  #
#===========================================#

x = torch.tensor([[1, 1], [1., 1.]], requires_grad=True)
out = x.pow(2).sum()
out.backward()
print(x.grad)

x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# Build a computational graph.
y = w * x + b    # y = 2 * x + 3

# Compute gradients.
y.backward()

# Print out the gradients.
print(x.grad)    # x.grad = 2
print(w.grad)    # w.grad = 1
print(b.grad) # b.grad = 1

#===========================================#
#                 Sample 2                  #
#              Basic autograd 2             #
#                  python3                  #
#===========================================#

# create tensor of shape(10,3)(10,2)
x = torch.randn(10, 3)
y = torch.randn(10, 2)
# Build a fully connected lawyer
linear = nn.Linear(3, 2)
print('W:', linear.weight)
print('b:', linear.bias)
# build loss function and optimizer(init)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)
# lr=learning rate

# Forward pass
pred = linear(x)

# Compute loss
loss = criterion(pred, y)
print('loss:', loss.item())
# Backward pass
loss.backward()
# Print grdient
print('dL/dW: ', linear.weight.grad)
print('dL/db: ', linear.bias.grad)

# Print 1-step gradient descent
optimizer.step()
pred = linear(x)
loss = criterion(pred, y)
optimizer.step()
print('loss after 1 step optimization: ', loss.item())

#===========================================#
#                 Sample 3                  #
#        Loading data from numpy            #
#                  python3                  #
#===========================================#

# Create a numpy array
x = np.array(([[1, 2], [3, 4]]))
print(x)
# Convert to a torch tensor
y = torch.from_numpy(x)
print(y)
# Convert to a numpy array
z = y.numpy()
print(z)

#============================================#
#                 Sample 4                   #
#              Input pipline                 #
#                  python3                   #
#============================================#

# Download and construct CIFAR-10 dataset.
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)

# Fetch one data pair (read data from disk).
image, label = train_dataset[0]
print (image.size())
print (label)

# Data loader (this provides queues and threads in a very simple way).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True)

# When iteration starts, queue and thread start to load data from files.
data_iter = iter(train_loader)

# Mini-batch images and labels.
images, labels = data_iter.next()

# Actual usage of the data loader is as below.
for images, labels in train_loader:
    # Training code should be written here.
    pass

#============================================#
#                 Sample 5                   #
#             Pretrained model               #
#                  python3                   #
#============================================#
"""
#download and load the pretrained ResNet-18
    
"""
resnet = torchvision.models.resnet18(pretrained=True)

# If you are want to finetune only the top layer of the model,set as below
for param in resnet.parameters():
    param.requires_grad = False
# Replace the top layer for finetuning
resnet.fc = nn.Linear(resnet.fc.in_features, 100)

# Forward pass
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print(outputs.size())

#===========================================#
#                 Sample 6                  #
#             Save and Load model           #
#                  python3                  #
#===========================================#
# Save and load the entire model
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# Save and load only the model parameters(recommenfed)
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))

#===========================================#
#                 Sample 7                  #
#            linear_regression              #
#                  python3                  #
#===========================================#
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# Hyper parameters
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# Toy dataser

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], [9.779],
                    [6.182], [7.59], [2.167], [7.042], [10.791], [5.313],
                    [7.997], [3.1]], dtype=np.float32)
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], [3.366],
                    [2.596], [2.53], [1.221], [2.827], [3.465], [1.65],
                    [2.904], [1.3]], dtype=np.float32)

# Linear regression model

model = nn.Linear(input_size, output_size)

# Loss and optimizer

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model

for epoch in range(num_epochs):
    # Convert numpy to torch tensors
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass and optimizer
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(epoch+1)%5 == 0:
        print('Epoch [{}/{}],loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Plot the result graph
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

# Save the model checkpoint

torch.save(model.state_dict(), 'model.ckpt')

#==========================================#
#               Sample 8                   #
#          logistic_regression             #
#             dataset MNIST                #
#                python3                   #
#==========================================#
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Hype-parameters

input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset (images and labels)

train_dataset = torchvision.datasets.MNIST(root='../../data',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)
test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())
# Data loader (input pipeline)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
# Logistic regression model

model = nn.Linear(input_size,num_classes)



# Loss and optimizer
# nn.CrossEntroyLoss() computes softmax internally

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # reshape images to (batch_size,input_size)
        images = images.reshape(-1, 28*28)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimizer
        optimizer.zero_grad()
        loss.backward
        optimizer.step()

        if (i+1)%100 == 0:
            print('Epoch [{}/{}],Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In the test pase,we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy of the model on 10000 test images: {} %'.format(100*correct/ total)) # 7%

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

#===========================================#
#                 Sample 9                  #
#          feedforward_neural_network       #
#               dataset MNIST               #
#                 python3                   #
#===========================================#
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset (images and labels)

train_dataset = torchvision.datasets.MNIST(root='../../data',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)
test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())
# Data loader (input pipeline)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# The model

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensor to the configured devices
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        #Forward paa
        outputs = model(images)
        loss=criterion(outputs, labels)

        # Backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%100 == 0:
            print('Epoch [{}/{}],Step [{}/{}],Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
# Test the model
# In the test pase,we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the model on 10000 test images: {} %'.format(100*correct/ total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

#============================================#
#                 Sample 10                  #
#               Basic ConvNet                #
#               dataset MNIST                #
#                  python3                   #
#============================================#
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset (images and labels)

train_dataset = torchvision.datasets.MNIST(root='../../data',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)
print(train_dataset)
test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())
# Data loader (input pipeline)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
# Convolutional neural network (two convoluation layers

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


# The model
model = ConvNet(num_classes).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        plt.show(images)
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss=criterion(outputs, labels)

        # Backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%100 == 0:
            print('Epoch [{}/{}],Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval() # eval mode
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(predicted)
    print('Test Accuracy of the model on the 10000 test images;{} %'.format(100*correct/total))

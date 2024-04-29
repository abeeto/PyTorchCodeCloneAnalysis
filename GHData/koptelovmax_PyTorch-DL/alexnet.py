import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F

import time
import timeit

import numpy as np
import matplotlib.pyplot as plt
#%%
# Define the original network class (as a reference):
class AlexNetOrig(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNetOrig, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4) # 96 @ (55 x 55)
        self.pool1 = nn.MaxPool2d(3, stride=2) # 96 @ (27 x 27)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2) # 256 @ (27 x 27)
        self.pool2 = nn.MaxPool2d(3, stride=2) # 256 @ (13 x 13)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1) # 384 @ (13 x 13)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1) # 384 @ (13 x 13)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1) # 256 @ (13 x 13)
        self.pool3 = nn.MaxPool2d(3, stride=2) # 256 @ (6 x 6)
        self.drop = nn.Dropout()
        self.fc1 = nn.Linear(256 * 6 * 6, 4096) # 1 x 4096
        self.fc2 = nn.Linear(4096, 4096) # 1 x 4096
        self.fc3 = nn.Linear(4096, num_classes) # 1 x num_classes

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool3(F.relu(self.conv5(x)))
        x = x.view(-1, 256 * 5 * 5) # flattening
        x = self.drop(x) # dropout to prevent overfitting
        x = F.relu(self.fc1(x))
        x = self.drop(x) # dropout to prevent overfitting
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Based on the reference class above define the one adapted to the CIFAR10 dataset:
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1) # 64 @ (16 x 16)
        self.pool1 = nn.MaxPool2d(2) # 64 @ (8 x 8)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1) # 192 @ (8 x 8)
        self.pool2 = nn.MaxPool2d(2) # 192 @ (4 x 4)
        self.conv3 = nn.Conv2d(192, 384, 3, padding=1) # 384 @ (4 x 4)
        self.conv4 = nn.Conv2d(384, 256, 3, padding=1) # 256 @ (4 x 4)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1) # 256 @ (4 x 4)
        self.pool3 = nn.MaxPool2d(2) # 256 @ (2 x 2)
        self.drop = nn.Dropout()
        self.fc1 = nn.Linear(256 * 2 * 2, 4096) # 1 x 4096
        self.fc2 = nn.Linear(4096, 4096) # 1 x 4096
        self.fc3 = nn.Linear(4096, num_classes) # 1 x num_classes

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool3(F.relu(self.conv5(x)))
        x = x.view(-1, 256 * 2 * 2) # flattening
        x = self.drop(x) # dropout to prevent overfitting
        x = F.relu(self.fc1(x))
        x = self.drop(x) # dropout to prevent overfitting
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
#%%
# Define a model:
net = AlexNet()

# Define optimizer:
#optimizer = torch.optim.SGD(params=net.parameters(), lr=0.01, 
#                            momentum=0.9, weight_decay=0.0005)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Objective function:
#criterion = nn.MSELoss()
#criterion = F.cross_entropy()

# Learning rate scheduler: multiply LR by 1 / 10 after every 30 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
#%%
# Load the data (the CIFAR10 dataset):
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100,
                                          shuffle=True, num_workers=0)

test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 
           'ship', 'truck')
#%%
# Image visualization:
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
#%%
# Training on GPU:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training on CPU:
#device = torch.device("cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

# move all modules to CUDA tensors:
net.to(device)
#%%
f_out = open('alexnet_log.txt','w')

# Train the network:
start = timeit.default_timer() # Initialize the timer to compute the time
start_cpu = time.process_time() # Initialize the cpu timer to compute time

# Numer of epochs:
epochs = 200

for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        
        # update learning rate:
        #lr_scheduler.step()
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #loss = criterion(outputs, labels)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            f_out.write('[%d, %5d] loss: %.3f \n' % 
                        (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    
print('Finished Training')

stop = timeit.default_timer() # stop the time counting
stop_cpu = time.process_time()  # stop the cpu time counting
print('Total time, sec: '+str(stop-start))
print('Time cpu, sec: '+str(stop_cpu-start_cpu))

f_out.write('Total time, sec: '+str(stop-start)+'\n')
f_out.write('Time cpu, sec: '+str(stop_cpu-start_cpu)+'\n')
f_out.close()

# Save our trained model:
PATH = './alex_net.pth'
torch.save(net.state_dict(), PATH)
#%%
# Test the network on the test data:
data_iter = iter(test_loader)
images, labels = data_iter.next()

# print images
imshow(torchvision.utils.make_grid(images))
images = images.to(device)
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# make predictions:
outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
#%%
# on the whole dataset:
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
#%%
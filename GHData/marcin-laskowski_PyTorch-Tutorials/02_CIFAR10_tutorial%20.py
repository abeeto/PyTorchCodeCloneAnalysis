# PyTorch version 0.4.0

'''
For this tutorial, we will use the CIFAR10 dataset. It has the classes: ‘airplane’, ‘automobile’,
‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of size
3x32x32, i.e. 3-channel color images of 32x32 pixels in size.


TRAINING IMAGE CLASSIFIER
We will do the following steps in order:
1. Load and normalizing the CIFAR10 training and test datasets using torchvision
3. Define a Convolution Neural Network
4. Define a loss function
5. Train the network on the training data
6. Test the network on the test data
'''

###################################################################################################
# 1.LOADING AND NORMALIZING CIFAR10

# for defining data
import torch
import torchvision
import torchvision.transforms as transforms
# for visualisation the dataset
import matplotlib.pyplot as plt
import numpy as np
# for defining the neural network
import torch.nn as nn
import torch.nn.functional as F
# for defining loss function
import torch.optim as optim


# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Define training set:
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
# Load the datasets
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
# usually for datasets too large to be loaded onto memory/ custom datasets you have to define
# a DataLoader class yourself but since the data is provided by pytorch itself we can simply use
# their function.
# batch_size  – how many samples per batch to load
# shuffle – set to True to have the data reshuffled at every epoch
# num_workers  – how many subprocesses to use for data loading. 0 means that the data will be
# loaded in the main process.

# Define testing set
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
# train = False, since this is our test set
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
# no need to shuffle the test set. It has never seen it before

# define classes:
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Visualise some images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()  # converting an image into a numpy array
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # we use transpose, because channels in torch have a different convention with respect with kind


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


###################################################################################################
# 2. DEFINING THE CONVOLUTION NEURAL NETWORK

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.conv2_drop= nn.Dropout2d(p=0.5) # can also define dropout for conv2d

        # conv2D produces an output of 16 channels (the multipliers are kernel size + padding)
        # there are 2 of them because we only defined 2 convolutional layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 classes hence, the output of the last layer has to be 10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # -1 essentially flattens the layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x= F.dropout(x,training=self.training) # defining the drop out
        x = self.fc3(x)
        return x


net = Net()

###################################################################################################
# 3. DEFINING A LOSS FUNCTION AND OPTIMIZER

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


###################################################################################################
# 4. TRAIN THE NETWORK

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


###################################################################################################
# 5. TESTING THE NETWORK ON THE TEST DATA

# We will check this by predicting the class label that the neural network outputs,
# and checking it against the ground-truth. If the prediction is correct,
# we add the sample to the list of correct predictions

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(images)

# The outputs are energies for the 10 classes. Higher the energy for a class,
# the more the network thinks that the image is of the particular class.

# Get the index of the highest energy:
_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# Performance of the network on the whole dataset
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# classes that performed well, and the classes that did not perform well:
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

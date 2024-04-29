"""
Implements a classifier based on a ResNet pre-trained on ImageNet
and tests it on CIFAR-10 and MNIST
Author: Ilia Smirnov (with some parts based on a CIFAR-10 tutorial
                      in the PyTorch documentation)
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda:0')
host = torch.device('cpu')
batch_size = 32

def CIFAR10_prep():
    #CIFAR-10 mean and stdev of images across R, G, B channels
    mean  = [0.491, 0.482, 0.447]
    stdev = [0.247, 0.243, 0.262]
    transform = transforms.Compose([ transforms.Resize(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, stdev) ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog',
               'frog', 'horse', 'ship', 'truck')
    return trainset, testset, classes

def MNIST_prep():
    mean  = [0.1307]
    stdev = [0.3081]
    transform = transforms.Compose([ transforms.Resize(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, stdev) ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=False, transform=transform)
    testset  = torchvision.datasets.MNIST(root='./data', train=False,
                                          download=False, transform=transform)
    classes = ('0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9')
    return trainset, testset, classes

trainset, testset, classes = CIFAR10_prep()

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=3)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=3)



class ResNetClassifier(nn.Module):
    """
    A classifier with a ResNet base and a smaller head
    """
    def __init__(self):
        super(ResNetClassifier, self).__init__()
        self.base = models.resnet18(pretrained=True)

        #Uncomment the line below for MNIST (switch number of input channels to 1 instead of 3)
        #nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)

        #Uncomment the two lines below to freeze the ResNet parameters:
        #for param in self.base.parameters():
        #    param.requires_grad = False

        self.head_input_size = 1000
        self.head = nn.Sequential( nn.Linear(self.head_input_size, 1024),
                                   nn.BatchNorm1d(1024),
                                   nn.ReLU(),
                                   nn.Linear(1024, 10) )

    def forward(self, x):
        x = F.relu(self.base(x))
        x = x.view(-1, self.head_input_size)
        x = self.head(x)
        return x

class ConvNetBlock(nn.Module):
    """
    A repeated architectural unit for the ConvNet classifier defined below
    """
    def __init__(self, in1, out1, in2, out2,
                 width1, width2,
                 stride1=1, stride2=1):
        super(ConvNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in1, out1, width1, stride=stride1)
        self.batch1 = nn.BatchNorm2d(out1)
        self.conv2 = nn.Conv2d(in2, out2, width2, stride=stride2)
        self.batch2 = nn.BatchNorm2d(out2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, stride=2)
        return x

class ConvNetClassifier(nn.Module):
    """
    A simpler Convolutional Neural Network architecture serving as 
    a baseline
    """
    def __init__(self):
        super(ConvNetClassifier, self).__init__()
        # input dim 224x224x3
        self.block1 = ConvNetBlock(3, 20, 20, 20, 5, 5)
        # 108x108x20
        self.block2 = ConvNetBlock(20, 20, 20, 20, 5, 5)
        # 50x50x20
        self.block3 = ConvNetBlock(20, 20, 20, 20, 5, 5)
        # 21x21x20
        self.block4 = ConvNetBlock(20, 20, 20, 20, 5, 4)
        # 7x7x20
        self.block5 = ConvNetBlock(20, 20, 20, 100, 4, 3)
        # 1x1x100
        self.ff_in  = 1 * 1 * 100
        self.ff1    = nn.Linear(self.ff_in, 1024)
        self.ff2    = nn.Linear(1024, 10)
        self.blocks = [self.block1, self.block2, self.block3, self.block4, self.block5]

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.view(-1, self.ff_in)
        x = self.ff1(x)
        x = F.relu(x)
        x = self.ff2(x)
        return x



# Set-up for ResNet classifier
classifier = ResNetClassifier()
optimizer = optim.SGD([
                           {'params': classifier.base.parameters(), 'lr': 0.001, 'momentum': 0.9},
                           {'params': classifier.head.parameters(), 'lr': 0.1, 'momentum': 0.9}
                      ])
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [9, 19])
#Finished ResNet setup


"""
# Set-up for ConvNet classifier
classifier = ConvNetClassifier()
optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [24])
#Finished ConvNet setup
"""

criterion = nn.CrossEntropyLoss()
classifier.to(device)

loss_text = open('plot/loss.txt', 'w')
train_acc_text = open('plot/train_acc.txt', 'w')
test_acc_text = open('plot/test_acc.txt', 'w')

#Evaluation
def evaluate(loader, loader_name):
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = classifier(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct/total
    print(loader_name + ' accuracy: %.1f %%' % (acc))
    return str(acc) + '\n'

# Training loop
epochs = 20
steps_per_epoch = len(trainset) / batch_size

train_acc_text.write(evaluate(trainloader, 'Training'))
test_acc_text.write(evaluate(testloader, 'Testing'))

for epoch in range(epochs):
    classifier.train()
    epoch_loss = 0.0
    for data in trainloader:
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = classifier(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    scheduler.step()
    print('[Epoch %d] loss: %.3f' %
           ( epoch + 1, epoch_loss / steps_per_epoch ))

    loss_text.write(str(epoch_loss / steps_per_epoch) + '\n')
    train_acc_text.write(evaluate(trainloader, 'Training'))
    test_acc_text.write(evaluate(testloader, 'Testing'))

loss_text.close()
train_acc_text.close()
test_acc_text.close()
print('Finished training')

PATH = './cifar_net.pth'
torch.save(classifier.state_dict(), PATH)

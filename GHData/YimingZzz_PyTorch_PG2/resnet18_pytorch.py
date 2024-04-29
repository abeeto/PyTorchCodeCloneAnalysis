# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 19:56:26 2018

@author: Yiming Zhou
"""

import torch

import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from tensorboardX import SummaryWriter

writer = SummaryWriter('D:\\UF2018_deeplearning_workshop\\assignment3_2')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root = 'D:\\UF2018_deeplearning_workshop\\data',
                                        train = True,
                                        download = True,
                                        transform = transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, 
                                          shuffle = True, num_workers = 0)

testset = torchvision.datasets.CIFAR10(root = 'D:\\UF2018_deeplearning_workshop\\data',
                                       train = False,
                                       download = True,
                                       transform = transform)

testloader = torch.utils.data.DataLoader(testset, batch_size = 64,
                                         shuffle = True, num_workers = 0)


class ResidualBlock (nn.Module):
    def __init__ (self, in_channels, out_channels, stride = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3,
                               stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3,
                               stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False),
                                          nn.BatchNorm2d(out_channels))
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layer_num, class_num = 10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace = True)
        #self.max_pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.layer1 = self.create_layer(block, 16, layer_num[0], stride = 1)
        self.layer2 = self.create_layer(block, 32, layer_num[1], stride = 2)
        self.layer3 = self.create_layer(block, 64, layer_num[2], stride = 2)
        self.ave_pool = nn.AvgPool2d(kernel_size = 8)
        self.fc = nn.Linear(64, 10)
        
    def create_layer(self, block, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)        
        layers = []
        for stride in strides:
            layers.append (block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        #out = self.max_pool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.layer4(out)
        out = self.ave_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
resnet_18 = ResNet(ResidualBlock, [3, 3, 3])

criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.SGD(resnet_18.parameters(), lr = lr, 
                            momentum = 0.9, weight_decay = 0.0001)

#train the network

train_iterations = []
test_iterations = []
train_loss = []
validation_loss = []
validation_acc= []
epoch_num = 10
for epoch in range(epoch_num):
    running_loss = 0.0
    if (epoch % 4 == 0 and epoch):
        lr /= 10
    print ("learning_rate: %f" % lr)
    for i, (images, labels) in enumerate(trainloader, 0):
        #running_loss = 0.0
        images = Variable(images)
        labels = Variable(labels)
        optimizer.zero_grad()
        outputs = resnet_18(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        #print (i)
        if (i + 1) % 100 == 0:
            print ("epoch [%d / %d], iter %d , loss %f" % (epoch + 1, epoch_num , i + 1, running_loss / (i+1)))
            #for training set
        counter_num = i+len(trainloader)*epoch  
        train_iterations.append(counter_num)
        train_loss.append(running_loss / (i+1))
        writer.add_scalar('training_loss', running_loss / (i+1), counter_num)

            
            
    correct = 0
    total = 0
    temp_loss = 0.0       
    for i, (images, labels) in enumerate(testloader):
        images = Variable(images)
        labels = Variable(labels)
        outputs = resnet_18(images)
        loss = criterion(outputs, labels)
        temp_loss += loss.data[0]
        counter_num = i+len(testloader)*epoch                    
        test_iterations.append(counter_num)
        validation_loss.append(temp_loss/(i + 1))    
        writer.add_scalar('validation_loss', temp_loss/(i + 1), counter_num)  

        
    for i, (images, labels) in enumerate(testloader):
        images = Variable(images)
        #labels = Variable(labels)
        outputs = resnet_18(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        counter_num = i+len(testloader)*epoch
        validation_acc.append(float(correct / total))    
        writer.add_scalar('validation_accuracy', float(correct / total), counter_num)
        
        


    print ('validation_loss : %f' % (temp_loss * 64 / 10000))

    print ("validation_accuracy: %f" %(correct / total))
    
    #for validation set



            

            
print ("Finish Training")
writer.close()

plt.figure(1)
plt.plot(train_iterations, train_loss)
#plt.xlabel('training iterations(6250 in one epoch)')
plt.ylabel('training_loss')
plt.title('Training Loss')
plt.show()

plt.figure(2)
plt.plot(test_iterations, validation_loss)
#plt.xlabel('training iterations(6250 in one epoch)')
plt.ylabel('validation_loss')
plt.title('Validation Loss')
plt.show()

plt.figure(3)
plt.plot(test_iterations, validation_acc)
#plt.xlabel('training iterations')
plt.ylabel('validation_accuracy')
plt.title('Validation Accuracy')
plt.show()

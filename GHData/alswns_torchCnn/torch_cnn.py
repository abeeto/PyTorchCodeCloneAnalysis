# -*- coding: utf-8 -*-
"""torch_cnn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vRF-9bKQ9bwlRx2MNMhfMoje-BEBCyQl
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

trainset = torchvision.datasets.MNIST('../mnist_data/',
                             download=True,
                             train=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,)) 
                             ])) 
    
testset = torchvision.datasets.MNIST("../mnist_data/", 
                             download=False,
                             train=False,
                             transform= transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307, ),(0.3081, ))
                           ]))
    
trainloader = torch.utils.data.DataLoader(trainset,                                         
                                         shuffle=True)


testloader = torch.utils.data.DataLoader(testset,                                         
                                         shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256,256)
        self.fc = nn.Linear(256,128)
      
        self.fc7 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(1,-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc(x))
        
     
        x = self.fc7(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.00040, momentum=0.9)



for epoch in range(10):  
    running_loss = 0.0
    for i, data in enumerate(trainloader): 
        inputs, labels = data
        optimizer.zero_grad()
   
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 600 == 0:    
            print('epoch : %d 진행도 : %.d%% loss : %.3f' %
                  (epoch + 1, i/600, running_loss / 2000))
            running_loss = 0.0
    correct = 0
    total = 0

    for data in testloader:
      images, labels = data
      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)
      total += 1
      correct += (predicted == labels).sum().item()

      print('%d / %d \n정확도: %lf %%' %(correct,total,
    100.0 * correct / total))
print('끝')

correct = 0
    total = 0

    for data in testloader:
      images, labels = data
      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)
      total += 1
      correct += (predicted == labels).sum().item()

      print('%d / %d \n정확도: %lf %%' %(correct,total,
    100.0 * correct / total))

!pip install git+https://github.com/demotomohiro/remocolab.git
import remocolab
remocolab.setupSSHD()


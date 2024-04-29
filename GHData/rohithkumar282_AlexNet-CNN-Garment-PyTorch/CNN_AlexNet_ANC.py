#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
import torch.nn as nn
import torchvision.datasets as dsets
from skimage import transform
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd;
import numpy as np;
from torch.utils.data import Dataset, DataLoader
from vis.utils import *
import random
import math

num_epochs = 5; #epochs
batch_size = 100;# batch size
learning_rate = 0.001;# learning rate

#Dataset modification
class FashionMNISTDataset(Dataset):
    
    def __init__(self, csv_file, transform=None):
        data = pd.read_csv(csv_file);
        self.X = np.array(data.iloc[:, 1:]).reshape(-1, 1, 28, 28)#.astype(float);
        self.Y = np.array(data.iloc[:, 0]);
        del data;
        self.transform = transform;
        
    def __len__(self):
        return len(self.X);
    
    def __getitem__(self, idx):
        item = self.X[idx];
        label = self.Y[idx];
        
        if self.transform:
            item = self.transform(item);
        
        return (item, label);
#get train and test data    
train_dataset = FashionMNISTDataset(csv_file='fashion-mnist_train.csv')
test_dataset = FashionMNISTDataset(csv_file='fashion-mnist_test.csv')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)
import matplotlib.pyplot as plt
#labels and explanation
labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt',
              7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'};
fig = plt.figure(figsize=(8,8))
columns = 4;
rows = 5;
#showing dataset
for i in range(1, columns*rows +1):
    img_xy = np.random.randint(len(train_dataset));
    img = train_dataset[img_xy][0][0,:,:]
    fig.add_subplot(rows, columns, i)
    plt.title(labels_map[train_dataset[img_xy][1]])
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()
#CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
cnn = CNN();
criterion = nn.CrossEntropyLoss();
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate);#initialise Adam Optimizer
losses = [];
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.float())
        labels = Variable(labels)#find predicted labels
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)#calculate loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (i+1) % 100 == 0:
            print ('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))
cnn.eval()
correct = 0
total = 0
#evaluate using test set and find accuracy
for images, labels in test_loader:
    images = Variable(images.float())
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Test Accuracy of the model on the 10000 test images: %.4f %%' % (100 * correct / total))
losses_in_epochs = losses[0::600]
plt.xlabel('Epoch #');
plt.ylabel('Loss');
plt.plot(losses);
plt.show();

#display one item and predict it
test_dataset1 = FashionMNISTDataset(csv_file='find_application.csv')
test_loader1 = torch.utils.data.DataLoader(dataset=test_dataset1,batch_size=batch_size,shuffle=True)
img_xy = np.random.randint(len(test_dataset1));
img = train_dataset[img_xy][0][0,:,:]
print('Input')
plt.title(labels_map[test_dataset1[img_xy][1]])
plt.axis('off')
plt.imshow(img, cmap='gray')
plt.show()
cnn.eval()
correct = 0
total = 0
#prediction
for images, labels in test_loader1:
    images = Variable(images.float())
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()      
print('Given input is ',labels_map[predicted.item()])


# In[ ]:





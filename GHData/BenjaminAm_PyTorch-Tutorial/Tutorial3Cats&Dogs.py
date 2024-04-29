# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 17:10:23 2018

@author: Benni
"""

import torch
import torchvision
from torchvision import transforms
from PIL import Image
from os import listdir
import random
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

# 16-43 Data preprocessing
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
transforms = transforms.Compose([
        transforms.Resize(256), # Verkleinern
        transforms.CenterCrop(256), # Quadratisch zuschneiden
        transforms.ToTensor(),
        normalize]) # ?

# TARGET: [isCat, isDog]
train_data_list = []
target_list = []
train_data = []
files = listdir("A:\\Datasets\\Cats&Dogs_Kaggle\\train\\")
for i in range(len(listdir("A:\\Datasets\\Cats&Dogs_Kaggle\\train\\"))):
    f = random.choice(files)
    files.remove(f)
    img = Image.open("A:\\Datasets\\Cats&Dogs_Kaggle\\train\\" + f)
    img_tensor = transforms(img) # (1,3,256, 256)
    train_data_list.append(img_tensor)
    isCat = 1 if 'cat' in f else 0
    isDog = 1 if 'dog' in f else 0
    target = [isCat, isDog]
    target_list.append(target)
    if len(train_data_list) >= 16:        
        train_data.append((torch.stack(train_data_list), target_list))
        train_data_list = []
        target_list = []
        print('Loaded batch ', len(train_data), 'of ', int(len(listdir('A:\\Datasets\\Cats&Dogs_Kaggle\\train\\'))/64))
        print('Percentage Done: ', 100*len(train_data)/int(len(listdir('A:\\Datasets\\Cats&Dogs_Kaggle\\train\\'))/64), '%')
        if len(train_data) > 100:
            break
        
class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5)
        self.conv3 = nn.Conv2d(12, 18, kernel_size=5)
        self.conv4 = nn.Conv2d(18, 24, kernel_size=5)
        self.fc1 = nn.Linear(14112, 1000)
        self.fc2 = nn.Linear(1000, 2)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = x.view(-1, 14112)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)

model = Netz()
        
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(epoch):
    model.train()
    batch_id = 0
    for data, target in train_data:
        target = torch.Tensor(target)
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        criterion = F.binary_cross_entropy
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(data), len(train_data),
                100. *batch_id / len(train_data), loss.data.item()))
        batch_id = batch_id + 1

def test():
    model.eval()
    files = listdir("A:\\Datasets\\Cats&Dogs_Kaggle\\test\\")
    f = random.choice(files)
    img = Image.open("A:\\Datasets\\Cats&Dogs_Kaggle\\test\\" + f)
    img_test_tensor = transforms(img)
    img_test_tensor.unsqueeze_(0)
    data = Variable(img_test_tensor)
    out = model(data)
    print(out.data.max(1, keepdim=True)[1].item())
    img.show()
    x = input("")
    
#for epoch in range(9):
 #   train(epoch)
test()
    

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 11:30:10 2018

@author: vprayagala2
"""
#%%
import os
import matplotlib.pyplot as plt

import torch
from torch import utils
from torch import nn , optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models
from torch import utils
#%%
image_size = (255,255)
crop_size=244
transform = transforms.Compose([transforms.Resize(image_size),
                                #transforms.Grayscale(),
                                transforms.CenterCrop(crop_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485,0.456,0.406),
                                                   (0.229,0.224,0.225))
                                ])
image_dir = "C:\\Data\\Images\\CatsNDogs"

train_dir = os.path.join(image_dir,"training_set")
test_dir = os.path.join(image_dir,"test_set")

train_data = datasets.ImageFolder(train_dir,transform=transform)
train_dataloader = utils.data.DataLoader(train_data,batch_size=64, shuffle = True)

test_data = datasets.ImageFolder(test_dir,transform=transform)
test_dataloader = utils.data.DataLoader(test_data,batch_size=64, shuffle = True)
#%%
images, labels = next(iter(train_dataloader))
#View one image
plt.imshow(images[0].view(crop_size,crop_size,-1))
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
model = models.resnet50(pretrained=True)
print(model)
#%%
#turn off gradients
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(nn.Linear(2048,512),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(512,2),
                           nn.LogSoftmax(dim=1)
                            )
model.fc = classifier

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.fc.parameters(),lr=0.003)

model.to(device)
#%%

epochs = 3
steps = 0
running_loss = 0

print_every =5
#%%
for epoch in range(epochs):
    for images, labels in train_dataloader:
        steps += 1
        
        images, labels = images.to(device) , labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model(images)
        loss = criterion(logps , labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval()
            test_loss = 0
            accuracy = 0
            
            for images, labels in test_dataloader:
                logps = model(images)
                loss = criterion(logps , labels)
                test_loss += loss.item()
                
                ps = torch.exp(logps)
                top_ps, top_class = ps.topk(1,dim=1)
                equality = top_class = labels.view(*top_class.shape)
                accuracy += torch.mean(equality.type(torch.FloatTensor))
                
            print("Epchs {}/{}..".format(epoch+1 , epochs),
              "Training Loss:{}..".format(running_loss/len(train_dataloader)),
              "Testing Loss:{}..".format(test_loss/len(test_dataloader)),
              "Accuracy:{}".format(accuracy/len(test_dataloader)))
        
        running_loss =0
        model.train()
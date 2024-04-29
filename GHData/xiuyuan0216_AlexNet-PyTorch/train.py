from parameters import *
from model import *

import os 
import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils import data 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms

if __name__ == 'main':
    
    alexnet = AlexNet(NUM_CLASSES).to(device)
    print(alexnet)
    
    dataset = datasets.ImageNet(root="./train",split="train",transforms=transforms.Compose([
        transforms.CenterCrop(IMAGE_DIM),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    dataloader = data.DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
    optimizer = optim.Adam(params=alexnet.parameters(), lr=0.0001)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    for epoch in range(1, NUM_EPOCHS+1):
        lr_scheduler.step()
        
        for imgs, classes in dataloader:
            optimizer.zero_grad()
            imgs, classes = imgs.to(device), classes.to(device)
            
            output = alexnet(imgs)
            loss = F.cross_entropy(output, classes)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                with torch.no_grad():
                    _, preds = torch.max(output, 1)
                    accuracy = torch.sum(preds == classes)
                    print("Epoch:{} \tLoss:{:.4f} \tAccuracy:{}".format(epoch, loss.item(), accuracy.item()))
            
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:22:36 2021

@author: mcgoug01
"""
import unet
import KiTs_Pytorch as kpt

import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import os


labels = 4
#change path variable to KiTS data folder ending in 'kits/data' containing many sub directories named 'case_00000, case_00001..' etc.
path = 'C:\\Users\\mcgoug01\\OneDrive - CRUK Cambridge Institute\\Python Scripts\\kits21\\kits21\\data'
kits = kpt.KiTS21_Data(path,n=10,num_class=labels)
kitsloader = DataLoader(dataset=kits,batch_size=3,shuffle=True)
model_loc = os.path.join(os.getcwd(),'unet')

if os.path.exists(model_loc): 
    model = torch.load(model_loc)
    print("Model loaded!")
else: model = unet.UNet(depth=5,in_channels=4,out_labels=labels)

costs=[]
opt = optim.Adam(model.parameters(),lr=0.0001)
#this line taken from https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("Total number of model parameters (in millions):",pytorch_total_params/1e6)
print("")
loss= nn.CrossEntropyLoss()
start = time.time()
epochs = 8
model.train()
for epoch in range(3):
    # bar = tqdm(kitsloader)
    for x,y in kitsloader:
        # bar.set_description("Training from KiTS")
        x = x.reshape(x.shape[0],x.shape[1],x.shape[2],-1)
        y = torch.transpose(y,1,3)
        pred = model(x.float())
        output = loss(pred, y)
        opt.zero_grad()
        output.backward()
        opt.step()
        costs.append(float(output))
        print("Loss: %.5f" % float(output))
    torch.save(model,model_loc)
        
plt.plot(costs)

index = 1
pred = model(x.reshape(x.shape[0],x.shape[1],x.shape[2],-1).float())
plt.subplot(131)
plt.xlabel("input image")
plt.imshow(x[index])
plt.subplot(132)
plt.xlabel("prediction")
plt.imshow(torch.transpose(pred,1,3)[index].argmax(-1))
plt.subplot(133)
plt.xlabel("target")
plt.imshow(torch.transpose(y,1,3)[index].argmax(-1))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import numpy as np
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms


class AlexNet(nn.Module):
    def __init__(self, n_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2)
                )
        self.avgPool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096), 
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, n_classes)
                )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgPool(x)
        print(x.shape)
        x = torch.flatten(x)
        print(x.shape)
        x = self.classifier(x)
        return x
        
        
model = AlexNet(1000)
model.load_state_dict(torch.load("data/alexnet-owt-4df8aa71.pth"))
model.eval()

import matplotlib.pyplot as plt
w=model.features[0].weight.cpu().detach().numpy()

#%%
#for i in range(w.shape[0]):
#    tmp = w[i, :, :, :].transpose((1, 2, 0))
#    plt.imshow(tmp)
#    plt.savefig("output/kernel_%04d.png" % i)
#    plt.close()
#

#%%
img_tmp = plt.imread("data/tiny-imagenet-200/test/images/test_2298.JPEG")
img_tmp = img_tmp[:, :, :3]*255
img_tmp = img_tmp.astype(np.uint8)

img2 = img_tmp.astype(np.float) / 255
img2 -= np.array([0.485, 0.456, 0.406])
img2 /= np.array([0.229, 0.224, 0.225])
img2 = torch.Tensor(img2)
print(img2.shape)
img2 = img2.permute(2, 0, 1)
img2 = img2.unsqueeze(0)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

from PIL import Image

imsize = 228
loader = transforms.Compose([transforms.ToPILImage(), transforms.Resize(imsize), transforms.ToTensor(), normalize])
img = loader(img_tmp).float()
img = img.unsqueeze(0)

#%%
with open('classes.txt','r') as inf:
    classes = eval(inf.read())

#%%
model.eval()
output = model(img)
index = torch.argmax(output)
#print(index.item(), classes[index.item()])

print("Top 5:")
_, indices = torch.topk(output, 5)
for i in indices.squeeze():
    print("\t- ", classes[i.item()])
    
#output = model(img2)
#index = torch.argmax(output)
#print(index.item(), classes[index.item()])

#%%
import torchvision.models as models
alexnet = models.alexnet(pretrained=True)
#alexnet.load_state_dict(torch.load("data/alexnet-owt-4df8aa71.pth"))

alexnet.eval()
output = alexnet(img)
index = torch.argmax(output)
#print(index.item(), classes[index.item()])
print("Top 5:")
_, indices = torch.topk(output, 5)
for i in indices.squeeze():
    print("\t- ", classes[i.item()])
    

resnet152 = models.resnet152(pretrained=True)
resnet152.eval()
output = resnet152(img)
print("Top 5:")
_, indices = torch.topk(output, 5)
for i in indices.squeeze():
    print("\t- ", classes[i.item()])
    

#output = alexnet(img2)
#index = torch.argmax(output)
#print(index.item(), classes[index.item()])
    
    
#%%
#imagenet_data = torchvision.datasets.CIFAR10('data/',
#                                              train=False,
#                                              transform=transforms.ToTensor(),
#                                              download=True)
#
#data_loader = torch.utils.data.DataLoader(imagenet_data,
#                                          batch_size=1,
#                                          shuffle=True)
#
#for img, label in data_loader:
#    print(label)
#    img_np = img[0, :, :, :].numpy().transpose((1, 2, 0))
#    
#    break
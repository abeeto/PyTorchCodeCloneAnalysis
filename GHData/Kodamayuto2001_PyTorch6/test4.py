#region __IMPORT__
import os
import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import numpy as np
import cv2
import sys
import time
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt
#endregion !__IMPORT__
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 1000)
        self.fc2 = torch.nn.Linear(1000, 2)
 
    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
 
        return F.log_softmax(x, dim=1)

class MyDataLoader:
    r"""
    trainRootDir = "path/to/train/"
    testRootDir = "path/to/test/"
    imgSize = 28
    batchSize = 128
    numWorkers = 3
        --numWorkers = the number of tagNames
    """
    def __init__(self,trainRootDir,testRootDir,imgSize,batchSize,numWorkers):
        #self.__my_collate_fn(batch=batchSize)
        self.__dataSet(trainRootDir=trainRootDir,testRootDir=testRootDir,imgSize=imgSize)
        self.__dataLoader(batchSize=batchSize,numWorkers=numWorkers)
        pass
    def __my_collate_fn(self,batch):
        pass 
    def __dataSet(self,trainRootDir,testRootDir,imgSize):
        self.trainData = torchvision.datasets.ImageFolder(root=trainRootDir,transform=transforms.Compose([transforms.Grayscale(),transforms.Resize((imgSize,imgSize)),transforms.ToTensor(),]))
        self.testData = torchvision.datasets.ImageFolder(root=testRootDir,transform=transforms.Compose([transforms.Grayscale(),transforms.Resize((imgSize,imgSize)),transforms.ToTensor(),]))
        pass
    def __dataLoader(self,batchSize,numWorkers):
        self.trainDataLoaders = torch.utils.data.DataLoader(self.trainData,batch_size=batchSize,shuffle=True,num_workers=numWorkers)
        self.testDataLoaders = torch.utils.data.DataLoader(self.testData,batch_size=batchSize,shuffle=True,num_workers=numWorkers)
        pass
    def imshow(self):
       print(type(self.trainDataLoaders))
       dataiter = iter(self.trainDataLoaders)
       img,labels = dataiter.next()
       img = torchvision.utils.make_grid(img)
       img = img /2 + 0.5
       npimg = img.numpy()
       plt.imshow(np.transpose(npimg,(1,2,0)))
       plt.show()
       pass
    def getDataLoaders(self):
        r"""
        RETURN
            --trainDataLoaders,
            --testDataLoaders
        """
        return {"train":self.trainDataLoaders,"test":self.testDataLoaders}
    pass

if __name__ == '__main__':
    net = Net()
    print(net.load_state_dict(torch.load('model.pt')))
    net.eval()

    mDataLoader = MyDataLoader(trainRootDir="Resources/train/",testRootDir="Resources/test/",imgSize=28,batchSize=10,numWorkers=1)
    loaders = mDataLoader.getDataLoaders()
    
    for i,data in enumerate(loaders["train"],0):
        inputs,label = data
        #print(inputs[0])
        #print(label[0])
        inputs = inputs.view(-1,28*28)
        output = net(inputs)
        pred = output.data.max(0,keepdim=True)[1]
        print(pred)
        #print(results)
        print(label)
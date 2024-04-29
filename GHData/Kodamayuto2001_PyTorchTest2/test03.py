import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
import numpy as np 
import pytorch_lightning as pl
from pytorch_lightning import Trainer

transform = transforms.Compose([transforms.ToTensor()])

train = torchvision.datasets.MNIST(root="Resources/",train=True,download=False,transform=transform)

train_val = torchvision.datasets.MNIST(root="Resources/",train=True,download=True,transform=transform)

test = torchvision.datasets.MNIST(root="Resources/",train=False,download=True,transform=transform)

n_train = int(len(train_val)*0.8)
n_val = len(train_val)-n_train

train,val=torch.utils.data.random_split(train_val,[n_train,n_val])

print(len(train))
print(len(val))
print(len(test))

class TrainNet(pl.LightningModule):
    @pl.data_loader
    def train_dataloader(self):
        return torch.utils.data.DataLoader(train,self.batch_size,shuffle=True)
    def training_step(self,batch,batch_nb):
        x,t = batch
        y = sekf.forward(x)
        loss = self.lossfun(y,t)
        results = {"loss":loss}
        return results

class ValidationNet(pl.LightningModule):
    @pl.data_loader
    def val_dataloader(self):
        return torch.utils.data.DataLoader(val,self.batch_size)

    def validation_step(self,batch,batch_nb):
        x,t = batch
        y = self.forward(x)
        loss = self.lossfun(y,t)
        y_label = torch.argmax(y,dim=1)
        acc = torch.sum(t==y_label)*1.0/len(t)
        results={"val_loss":loss,"val_acc":acc}
        return results
            
    def validation_end(self,outputs):
        avg_loss = torch.stack([x["val_loss"]for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"]for x in outputs]).mean()
        retults = {"val_loss":avg_loss,"val_acc":avg_acc}
        return results

class TestNet(pl.LightningModule):
    @pl.data_loader
    def test_dataloader(self):
        return torch.utils.data.DataLoader(test,self.batch_size)
    
    def test_step(self,batch,batch_nb):
        x,t = batch
        y = self.forward(x)
        loss = self.lossfun(y,t)
        y_label = torch.argmax(y,dim=1)
        acc = torch.sum(t==y_label)*1.0/len(t)
        results = {"test_loss":loss,"test_acc":acc}
        return results
    
    def test_end(self,outputs):
        avg_loss = torch.stack([x["test_loss"]for x in outputs]).mean()
        avg_acc = torch.stack([x["test_acc"]for x in outputs]).mean()
        results = {"test_loss":avg_loss,"test_acc":avg_acc}
        return results

class Net(TrainNet,ValidationNet,TestNet):
    def __init__(self,input_size=784,hidden_size=100,output_size=10,batch_size=256):
        super(Net,self).__init__()
        self.batch_size = batch_size
        #使用する層の宣言
        self.conv = nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,stride=1,padding=1)
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)
    
    def lossfun(self,y,t):
        return F.cross_entropy(y,t)
    
    def cofigure_optimizers(self):
        return torch.optim.SGD(self.parameters(),lr=0.01)
    
    def forward(self,x):
        x = self.conv(x)
        x = F.max_pool2d(x,2,2)
        x = x.view(x,size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(X)
        return x

net = Net()
trainer = Trainer(gpus=1)
trainer.fit(net)

    
    
    
    
    
    
    
    
    
    
    

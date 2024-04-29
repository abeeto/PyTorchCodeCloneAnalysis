import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning import Trainer
from torchvision import transforms
import torch
import torchvision

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


class Net(pl.LightningModule):
    def __init__(self, input_size=4, hidden_size=4, output_size=3, batch_size=10):
        super(Net, self).__init__()
        self.L1 = nn.Linear(input_size, hidden_size)
        self.L2 = nn.Linear(hidden_size, output_size)
        self.batch_size = batch_size
        
        self.bn = nn.BatchNorm1d(input_size)
        
    def forward(self, x):
        x = self.L1(x)
        x = F.relu(x)
        x = self.L2(x)
        return x
    
    def lossfun(self, y, t):
        return F.cross_entropy(y, t)
        #return nn.CrossEntropyLoss(self, y, t)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)
        
    @pl.data_loader
    def train_dataloader(self):
        return torch.utils.data.DataLoader(train, self.batch_size, shuffle=True)
    
    def training_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        results = {'loss': loss}
        return results
net = Net() # 学習モデルのインスタンス化
trainer = Trainer() # 学習用のインスタンス化と学習の設定
trainer.fit(net) # 学習ループ実行
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch 
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torchvision import transforms



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testset = CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class NeuralNetwork(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(3, 10, (3, 3), padding=0, stride=2)
        self.layer2 = nn.Conv2d(10, 10, (3, 3), padding=0, stride=2)
        self.layer3 = nn.Conv2d(10, 10, (3, 3), padding=0, stride=1)
        self.layer4 = nn.Linear(250, 128)
        self.layer5 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = x.view(-1, 250)
        x = F.relu(self.layer4(x))
        x = F.log_softmax(self.layer5(x))
        
        return x
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.nll_loss(y_pred, y)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        self.log('val_acc', pl.metrics.functional.classification.accuracy(y_pred, y), prog_bar=True)
        loss = F.nll_loss(y_pred, y)
        return loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        return pl.metrics.functional.classification.accuracy(y_pred, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
        

model = NeuralNetwork()
trainer = Trainer(max_epochs=5)

trainer.fit(model, trainloader, testloader)


#sprawdzenie szejpu
X, y = next(iter(testloader))
model(X).view(32, -1).shape
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
from torchvision.datasets import MNIST
from torchvision import transforms

transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])

mnist_train = MNIST('data', train=True, download=True, transform=transform)
mnist_test = MNIST('data', train=False, download=True, transform=transform)


mnist_train = torch.utils.data.DataLoader(mnist_train, batch_size=256, shuffle=True)
mnist_test = torch.utils.data.DataLoader(mnist_test, batch_size=256)

X, y = next(iter(mnist_train))

class NeuralNetwork(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(1, 20, kernel_size=(3, 3), padding=0)
        self.layer2 = nn.Conv2d(20, 25, kernel_size=(3, 3), padding=0)
        self.layer3 = nn.Conv2d(25, 15, kernel_size=(3, 3), padding=0)
        self.layer4 = nn.Linear(7260, 512)
        self.layer5 = nn.Linear(512, 10)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = x.view(-1, 7260)
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
#model.layer3(model.layer2(model.layer1(X))).view(32, -1).size()

trainer.fit(model, mnist_train, mnist_test)

X_test, y_test = mnist_test

# Audience: Users who need to train a model without coding their own training loops.
# https://pytorch-lightning.readthedocs.io/en/latest/model/train_model_basic.html

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import torch.utils.data as data
from torchvision import datasets
import torchvision.transforms as transforms

import pytorch_lightning as pl

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)

# Define the LightningModule
# The training_step defines how the nn.Modules interact together.
# In the configure_optimizers define the optimizer(s) for your models.

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.save_hyperparameters() # save all the hyperparameters pass to init. 

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)

# Load data sets

transform = transforms.ToTensor()
# Train and validation set 

train_set = datasets.MNIST(root=os.getcwd(), download=False, train=True, transform=transform)
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)
# dataloader : combines a dataset & and a sampler : 
# and provides an iterator over the given dataset. 
train_loader = DataLoader(train_set)
valid_loader = DataLoader(valid_set)
# Add a test set.
test_set = datasets.MNIST(root=os.getcwd(), download=False, train=False, transform=transform)

autoencoder = LitAutoEncoder(Encoder(), Decoder())

trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(autoencoder, train_loader, valid_loader)
trainer.test(autoencoder, dataloaders=DataLoader(test_set))


from torch.nn import *
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import wandb, os

PROJECT_NAME = "Intel-Image-Classification-Learning-PyTorch-Lightning"
criterion = MSELoss()


class Model(Module):
    def __init__(self):
        super().__init__()
        self.activation = ReLU()
        self.linear1 = Linear(3 * 5 * 5, 256)
        self.linear2 = Linear(256, 512)
        self.linear3 = Linear(512, 1024)
        self.linear4 = Linear(1024, 10)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        preds = self.activation(self.linear1(images))
        preds = self.activation(self.linear2(preds))
        preds = self.activation(self.linear3(preds))
        preds = self.activation(self.linear4(preds))
        loss = criterion(preds, labels)
        wandb.log({"Loss": loss.item()})
        return {"train_loss": loss.item()}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        preds = self.activation(self.linear1(images))
        preds = self.activation(self.linear2(preds))
        preds = self.activation(self.linear3(preds))
        preds = self.activation(self.linear4(preds))
        loss = criterion(preds, labels)
        wandb.log({"Val Loss": loss.item()})
        return {"val_loss": loss.item()}

    def train_dataloader(self):
        dataset = torchvision.datasets.MNIST(
            "./data/",
            download=True,
        )
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        return data_loader

    def val_dataloader(self):
        dataset = torchvision.datasets.MNIST("./data/", download=True, train=False)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        return data_loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


model = Model()
wandb.init(project=PROJECT_NAME, name="baseline")
trainer = Trainer()
trainer.fit(model)
wandb.finish()

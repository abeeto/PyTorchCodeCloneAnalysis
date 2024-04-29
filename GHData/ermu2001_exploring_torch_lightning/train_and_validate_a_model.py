import os
import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__=='__main__':
    # model to train and dataset-dataloader for training
    from model import a_very_simple_encoder,a_very_simple_decoder
    # Load data sets
    train_set = datasets.MNIST(root="MNIST", download=True, train=True)
    
    # to split the train set into real-training set and validation set
    # use 20% of training data for validation
    train_set_size = int(len(train_set) * 0.8)
    valid_set_size = len(train_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)
    test_set = datasets.MNIST(root="MNIST", download=True, train=False)  
    # pl.LightningModule
    

    train_loader = DataLoader(train_set)
    valid_loader = DataLoader(valid_set)
    test_loader = DataLoader(test_set)
    
    autoencoder = LitAutoEncoder(a_very_simple_encoder,a_very_simple_decoder)
    trainer = pl.Trainer()
    DONE_TRAINING = False
    if DONE_TRAINING:
        # test the model
        trainer.test(model=autoencoder, dataloaders=DataLoader(test_loader))
    else:
        # train model
        trainer.fit(model=autoencoder, train_dataloaders=train_loader,eval_dataloaders=valid_loader)
    
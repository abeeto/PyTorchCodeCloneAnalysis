import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl


class LitModel(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.l0 = torch.nn.Linear(28 * 28, 2**5)
        self.l1 = torch.nn.Linear(2**5, 10)

    def forward(self, x):
        x = F.relu(self.l0(x.view(x.size(0), -1)))
        return self.l1(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def prepare_data(self):
        # download only
        train_dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        test_dataset = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

        # train/val split
        train_dataset, val_dataset = random_split(train_dataset, [55000, 5000])

        # assign to use in dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=64, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=64, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=64, num_workers=4)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        correct = (y == y_hat.argmax(1)).float()
        return {'val_loss': F.cross_entropy(y_hat, y), 'correct': correct}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        acc = torch.cat([x['correct'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        correct = (y == y_hat.argmax(1)).float()
        return {'test_loss': F.cross_entropy(y_hat, y), 'correct': correct}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        acc = torch.cat([x['correct'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': acc}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}


model = LitModel({})

early_stopping = pl.callbacks.EarlyStopping('val_loss', patience=2)

# most basic trainer, uses good defaults
trainer = pl.Trainer(gpus=0, num_nodes=1, early_stop_callback=early_stopping)
trainer.fit(model)

# run test set
trainer.test()

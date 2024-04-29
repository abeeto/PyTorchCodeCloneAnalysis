import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl


class Model(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()

        # Build layers

        # Set hparams

    # Model computation and steps
    def model_base(self, x):
        pass

    def forward(self, x):
        out = self.model_base(x)
        # Replace with act. function
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self.model_base(x), y)
        self.log("train-loss",
                 loss,
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self.forward_base(x), y)
        self.log("val-loss", loss, on_epoch=True, on_step=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self.forward_base(x), y)
        self.log("test-loss", loss, on_epoch=True)

    # Network configuration
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=self.step_size,
                                                 gamma=self.lr_decay)
        return [optimizer], [lr_scheduler]

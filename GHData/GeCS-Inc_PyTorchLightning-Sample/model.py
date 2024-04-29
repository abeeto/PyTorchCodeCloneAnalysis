from utils import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from pytorch_lightning.core import LightningModule

import torchvision.models as models
from efficientnet_pytorch import EfficientNet

MODELS = [
    "vgg16", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnet101_v2",
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8'
]


class FineTuningModel(LightningModule):

    def __init__(self, env):
        super().__init__()
        self.save_hyperparameters()

        if env == {}:
            # save.hyperparameters()を行っていなかったため
            from train import env

        if type(env) == dict:
            env = EasyDict(env)

        self.env = env

        assert env.base_model in MODELS

        if env.base_model == "vgg16":
            self.model = models.vgg16(pretrained=True)
            self.model = nn.Sequential(
                *list(self.model.children())[:-2])
            fc_in_features = 512

        if env.base_model.startswith("resnet"):
            self.model = getattr(models, env.base_model)(pretrained=True)
            fc_in_features = self.model.fc.in_features
            self.model = nn.Sequential(*list(self.model.children())[:-2])

        if env.base_model.startswith("efficientnet"):
            self._model = EfficientNet.from_pretrained(
                env.base_model, include_top=False)
            fc_in_features = self._model._fc.in_features
            self.model = self._model.extract_features

        self.dropout = nn.Dropout(env.dropout_rate)
        self.fc = nn.Linear(fc_in_features, env.num_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    def training_step(self, batch, batch_idx):
        # forward pass
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {'val_loss': val_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {'test_loss': test_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = sum([x['n_correct_pred'] for x in outputs]) / \
            sum(x['n_pred'] for x in outputs)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': val_acc}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc = sum([x['n_correct_pred'] for x in outputs]
                       ) / sum(x['n_pred'] for x in outputs)
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': test_acc}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(
            self.parameters(), lr=self.env.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

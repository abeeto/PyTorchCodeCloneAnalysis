# Using PreTrain models
# https://pytorch-lightning.readthedocs.io/en/latest/advanced/transfer_learning.html

# Example : ImageNet

import torch
from torch import nn
import torchvision.models as models

import pytorch_lightning as pl

class ImagenetTransferLearning(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = 10
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)


model = ImagenetTransferLearning()
trainer = pl.Trainer()
trainer.fit(model)
import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import os
from torchvision import datasets, transforms
from torch.optim import Adam


# transforms
# prepare transforms standard to MNIST
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])

# data
mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
mnist_train = DataLoader(mnist_train, batch_size=64)

# class LitMNIST(LightningModule):

#   def __init__(self):
#     super().__init__()

#     # mnist images are (1, 28, 28) (channels, width, height)
#     self.layer_1 = torch.nn.Linear(28 * 28, 128)
#     self.layer_2 = torch.nn.Linear(128, 256)
#     self.layer_3 = torch.nn.Linear(256, 10)

#   def forward(self, x):
#     batch_size, channels, width, height = x.size()

#     # (b, 1, 28, 28) -> (b, 1*28*28)
#     x = x.view(batch_size, -1)
#     x = self.layer_1(x)
#     x = F.relu(x)
#     x = self.layer_2(x)
#     x = F.relu(x)
#     x = self.layer_3(x)

#     x = F.log_softmax(x, dim=1)
#     return x

# net = LitMNIST()
# x = torch.randn(1, 1, 28, 28)
# out = net(x)
# print(out)

# class LitMNIST(LightningModule):

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = F.nll_loss(logits, y)
#         return loss
class LitMNIST(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss
# for gpu 
model = LitMNIST()
trainer = Trainer(gpus=1)
trainer.fit(model, train_loader)
# for cpu
# model = LitMNIST()
# trainer = Trainer()
# trainer.fit(model, mnist_train)

optimizer = Adam(LitMNIST().parameters(), lr=1e-3)


# for epoch in epochs:
#     for batch in data:
#         # ------ TRAINING STEP START ------
#         x, y = batch
#         logits = model(x)
#         loss = F.nll_loss(logits, y)
#         # ------ TRAINING STEP END ------

#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()


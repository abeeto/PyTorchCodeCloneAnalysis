import torch
import torch.nn as nn

# from dqn.modules import Resize
from dqn import Resize


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.criterion = nn.CrossEntropyLoss()
        self.model = nn.Sequential(
            nn.Conv2d(2, 32, 3),
            nn.LeakyReLU(.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2, 2),
            Resize(-1, 192),
            nn.Linear(192, 64),
            nn.LeakyReLU(.1),
            nn.Linear(64, 5),
            nn.LeakyReLU(.1)
        )

    def forward(self, x):
        # x = self.flatten(x)
        # logits = self.linear_relu_stack(x)
        # return logits
        return torch.softmax(self.model(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        # x = self.flatten(x)
        return self.criterion(self.model(x), torch.argmax(y))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.forward(x).argmax(0), y.argmax(0)).float())

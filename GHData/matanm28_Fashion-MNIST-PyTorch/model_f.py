import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim

from model_base import ModelBase


class ModelF(ModelBase):
    def __init__(self, image_size: int, lr: float):
        super(ModelF, self).__init__(image_size, lr)
        self.hidden_layer_0 = nn.Linear(image_size, 128)
        self.hidden_layer_1 = nn.Linear(128, 64)
        self.hidden_layer_2 = nn.Linear(64, 10)
        self.hidden_layer_3 = nn.Linear(10, 10)
        self.hidden_layer_4 = nn.Linear(10, 10)
        self.output_layer = nn.Linear(10, 10)
        self.sigmoid = nn.Sigmoid()

    @property
    def optimizer(self):
        return optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, data: Tensor):
        data = data.view(-1, self.image_size)
        data = self.sigmoid(self.hidden_layer_0(data))
        data = self.sigmoid(self.hidden_layer_1(data))
        data = self.sigmoid(self.hidden_layer_2(data))
        data = self.sigmoid(self.hidden_layer_3(data))
        data = self.sigmoid(self.hidden_layer_4(data))
        data = self.sigmoid(self.output_layer(data))
        return F.log_softmax(data, dim=1)

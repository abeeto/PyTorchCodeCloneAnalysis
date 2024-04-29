import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim

from model_base import ModelBase


class ModelA(ModelBase):
    def __init__(self, image_size: int, lr: float):
        super(ModelA, self).__init__(image_size, lr)
        self.hidden_layer_0 = nn.Linear(image_size, 100)
        self.hidden_layer_1 = nn.Linear(100, 50)
        self.output_layer = nn.Linear(50, 10)

    @property
    def optimizer(self):
        return optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, data: Tensor):
        data = data.view(-1, self.image_size)
        data = F.relu(self.hidden_layer_0(data))
        data = F.relu(self.hidden_layer_1(data))
        data = F.relu(self.output_layer(data))
        return F.log_softmax(data, dim=1)

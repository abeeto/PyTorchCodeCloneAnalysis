import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim
from typing import Dict
from model_base import ModelBase


class ModelC(ModelBase):
    def __init__(self, image_size: int, lr: float, additional_params: Dict):
        super(ModelC, self).__init__(image_size, lr)
        self.hidden_layer_0 = nn.Linear(image_size, 100)
        self.dropout_layer_0 = nn.Dropout(additional_params['dropout'][0])
        self.hidden_layer_1 = nn.Linear(100, 50)
        self.dropout_layer_1 = nn.Dropout(additional_params['dropout'][1])
        self.output_layer = nn.Linear(50, 10)

    @property
    def optimizer(self):
        return optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, data: Tensor):
        data = data.view(-1, self.image_size)
        data = F.relu(self.hidden_layer_0(data))
        data = self.dropout_layer_0(data)
        data = F.relu(self.hidden_layer_1(data))
        data = self.dropout_layer_1(data)
        data = F.relu(self.output_layer(data))
        return F.log_softmax(data, dim=1)

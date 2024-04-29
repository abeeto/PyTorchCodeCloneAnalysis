import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

class MlModel:

    def __init__(self, model):
        self.model = model

    def manifest(self):
        # if null throw error
        return self.model
        
    def loss_func(self, batch_outputs, batch_labels):
        loss = nn.CrossEntropyLoss()
        return loss(batch_outputs, batch_labels)
        # batch_size = batch_outputs.size()[0]
        # return -torch.sum(batch_outputs[range(batch_size), batch_labels]) / batch_size

    def optimizer(self, params=None):
        # return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9) 
        return optim.Adam(self.model.parameters(), lr=0.0001)
        
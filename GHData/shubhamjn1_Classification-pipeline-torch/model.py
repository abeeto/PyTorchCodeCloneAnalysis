"""
Incorporates all the model arhcitectures build for training
"""
import torch
import torch.nn as nn


class classifierDNN(nn.Module):
    """DNN architecture for training neural nets on the data

    Args:
        nn ([object]): [NN module from torch]
    """
    def __init__(self, num_features, num_class):
        # super.__init__()
        super(classifierDNN, self).__init__()

        self.layer1 = nn.Linear(num_features, 512)
        self.layer2 = nn.Linear(512, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.layer1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x

def classifierCNN(self, parameter_list):
    """
    docstring
    """
    raise NotImplementedError
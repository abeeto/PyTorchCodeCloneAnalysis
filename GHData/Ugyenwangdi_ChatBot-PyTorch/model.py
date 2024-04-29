# PyTorch model and training

import torch
import torch.nn as nn

# Create a new class for our model
class NeuralNet(nn.Module):  # derived from nn.Module
    def __init__(self, input_size, hidden_size, num_classes):  # Feed forward neural net
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()  #Activation function

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)  # activation function
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)  # third linear layout
        # no activation and no softmax 

        return out

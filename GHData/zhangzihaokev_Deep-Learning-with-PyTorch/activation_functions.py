#  Recall that an activation function applies a non-linear transformation to the input of a neuron and determines whether or not that neuron should fire. If we use linear transformations, our network is essentially a stack of linear regressions. Using non-linear transformations is better suited for making more complex predictions. 
# Popular activation functions:
# 1. Step function (not used in practice, but good demonstration of theory)
# 2. Sigmoid (outputs values between 0 and 1, good for last layer in binary classification problems)
# 3. TanH (scaled & shifted sigmoid, outs values between -1 and 1, good for hidden layers)
# 4. ReLU (most popular choice, outputs 0 for negative values and its normal value for positive, if you don't know what to use, use ReLU)
# 5. Leaky ReLU (slightly improved ReLU, scales the input by very small epsilon for negative values, tries to solve the vanishing gradient problem since the input is 0 the gradient is 0 and the neuron won't learn anything)
# 6. Softmax (outputs values between 0 and 1, good for last layer in multiclass classification problems)

from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

# Option 1: create nn modules
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


# Option 2: call activation functions directly in forward pass
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(input_size, 1)
    
    # can use from nn or torch api
    def forward(self, x):
        out = nn.ReLU(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch import nn


class MLP(nn.Module):
    
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super(MLP, self).__init__()
        self.num_hid = len(n_hidden)
        self.layers = list()
        for i in range(self.num_hid):
            name = "hidden_layer" + str(i)
            linear = None
            if i == 0:
                linear = nn.Linear(n_inputs, n_hidden[i])
                self.add_module(name, linear)
            else:
                linear = nn.Linear(n_hidden[i - 1], n_hidden[i])
                self.add_module(name, linear)
            self.layers.append(linear)
        self.output_layer = nn.Linear(n_hidden[-1], n_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None


    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        linear_input = x
        for layer in self.layers:
            linear_input = layer(linear_input).clamp(min=0)
        linear_output = self.output_layer(linear_input)
        softmax = nn.Softmax()
        forward_output = softmax(linear_output)
        return forward_output

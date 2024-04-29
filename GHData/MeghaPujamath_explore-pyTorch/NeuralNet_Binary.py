import torch
import torch.nn as nn


class Binary_NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Binary_NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)

        y_pred = torch.sigmoid(out)
        return y_pred


# Define model
model = Binary_NeuralNet(input_size=28*28, hidden_size=5)
loss = nn.BCELoss()

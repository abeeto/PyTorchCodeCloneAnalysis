import torch
import torch.nn as nn


class basic_MLP_with_PE(nn.Module):
    def __init__(self, hidden_sizes):
        super(basic_MLP_with_PE, self).__init__()
        hidden_sizes = [32 * 32 * 3] + hidden_sizes + [100]
        layers = [nn.Flatten()]
        for hidden_before, hidden_after in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            layers.append(nn.Linear(hidden_before, hidden_after))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers[:-1])
        self.register_parameter("pe", nn.parameter.Parameter(torch.zeros(1, 3, 32, 32)))

    def forward(self, x):
        batch_size = x.shape[0]
        x += self.pe.repeat((batch_size, 1, 1, 1))
        return self.layers(x)


class basic_MLP_without_PE(nn.Module):
    def __init__(self, hidden_sizes):
        super(basic_MLP_without_PE, self).__init__()
        hidden_sizes = [32 * 32 * 3] + hidden_sizes + [100]
        layers = [nn.Flatten()]
        for hidden_before, hidden_after in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            layers.append(nn.Linear(hidden_before, hidden_after))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers[:-1])

    def forward(self, x):
        return self.layers(x)

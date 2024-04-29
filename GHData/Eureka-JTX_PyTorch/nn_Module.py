import torch
from torch import nn


class Eureka(nn.Module):
    def __init__(self):
        super() .__init__()

    def forward(self, input):
        output = input + 1
        return output


eu = Eureka()
x = torch.tensor(1.0)
output = eu(x)
print(output)

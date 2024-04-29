from nn.conv import Conv2d
from nn.module import Module


class DemoConv(Module):
    def __init__(self):
        super().__init__()

        self.conv = Conv2d(1, 2, 3)

    def forward(self, x):
        return self.conv(x)

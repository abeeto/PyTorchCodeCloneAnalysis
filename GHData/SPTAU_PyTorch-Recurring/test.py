import torch
import torch.nn as nn
from torch import Tensor


class test1(nn.Sequential):
    def __init__(self, num_input_features: int = 32, num_output_features: int = 64) -> None:
        super().__init__()
        self.Conv = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False),
        )
        self.AvgPool = nn.AvgPool2d(kernel_size=2, stride=2)


class test2(nn.Module):
    def __init__(self, num_input_features: int = 32, num_output_features: int = 64) -> None:
        super().__init__()
        self.Conv = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False),
        )
        self.AvgPool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor):
        output = self.Conv(x)
        output = self.AvgPool(output)
        return output


x = torch.randn(64, 32, 224, 224)

model1 = test1()
model2 = test2()

y1 = model1(x)
print(y1.size())

y2 = model2(x)
print(y2.size())

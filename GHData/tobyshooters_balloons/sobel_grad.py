import torch
import torch.nn as nn

class SobelGrad(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        kernel = torch.Tensor([[[ [1, 0, -1], [2, 0, -2], [1, 0, -1] ]]])

        # X-direction sobel operator in the form of convolution
        self.conv_x = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        self.conv_x.weight = torch.nn.Parameter(kernel)
        self.conv_x = self.conv_x.to(device)

        # Y-direction sobel operator in the form of convolution
        self.conv_y = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        self.conv_y.weight = torch.nn.Parameter(kernel.transpose(-1, -2))
        self.conv_y = self.conv_y.to(device)

    def forward(self, pred):
        grad_x = self.conv_x(pred)
        grad_y = self.conv_y(pred)
        return torch.pow(grad_x, 2) + torch.pow(grad_y, 2)


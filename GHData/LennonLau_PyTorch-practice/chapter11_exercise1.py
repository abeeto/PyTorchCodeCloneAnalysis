import torch


class ConstantScaling(torch.nn.Module):
    def __init__(self, in_channels):
        super(ConstantScaling, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        y = self.relu(self.conv1(x))
        y = self.conv2(y)

        return self.relu(0.5*x + 0.5*y)


class ConvShortcut(torch.nn.Module):
    def __init__(self, in_channels):
        super(ConvShortcut, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        y = self.relu(self.conv2(x))
        y = self.conv3(y)

        x = self.conv1(x)

        return self.relu(x+y)

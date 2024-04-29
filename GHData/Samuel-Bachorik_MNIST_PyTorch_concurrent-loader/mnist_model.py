from torch import nn
import torch
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.activation2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.activation3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.activation4 = nn.ReLU()

        self.linear1 = nn.Linear(128 * 5 * 5, 10)
        self.soft = nn.LogSoftmax(dim=1)

    def forward(self, xb):
        xb = self.conv1(xb)
        xb = self.activation1(xb)

        xb = self.conv2(xb)
        xb = self.activation2(xb)

        xb = self.conv3(xb)
        xb = self.activation3(xb)
        xb = self.conv4(xb)
        xb = self.activation4(xb)

        xb = xb.reshape(-1, 128 * 5 * 5)

        xb = self.linear1(xb)
        xb = self.soft(xb)

        return xb
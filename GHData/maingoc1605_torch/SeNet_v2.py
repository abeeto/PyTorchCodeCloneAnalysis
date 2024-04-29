from torch import nn
import torch

class SeNet(nn.Module):
    def __init__(self,reduction, channel):
        super(SeNet, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        y = self.squeeze(input)
        y = self.excitation(y)
        output = input * (y.expand_as(input))
        return output
    
if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    bn, c, _, _ = input.size()
    se = SeNet(channel=c, reduction=8)
    output = se(input)
    print(output.shape)

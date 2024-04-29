import torch
import torchvision
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class Eureka(nn.Module):
    def __init__(self):
        super(Eureka, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


writer = SummaryWriter("logs")
eu = Eureka()
step = 0
for data in dataloader:
    imgs, targets = data
    output = eu(imgs)
    print(output.shape)
    # torch.Size([64, 6, 32, 32])
    writer.add_images("input", imgs, step)
    # torch.Size([64, 6, 30, 30])
    output=torch.reshape(output,(-1,3,30,30))
    writer.add_images("output", output, step)
    step += 1

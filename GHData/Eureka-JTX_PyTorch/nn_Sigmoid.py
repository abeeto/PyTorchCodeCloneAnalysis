import torchvision
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import Sigmoid
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class Eureka(nn.Module):
    def __init__(self):
        super(Eureka, self).__init__()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output


writer = SummaryWriter("logs")
eu = Eureka()
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = eu(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root = "./dataset", train=True,
                                       transform=torchvision.transforms.ToTensor(),download=False)
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

class Myclass(nn.Module):
    def __init__(self):
        super(Myclass,self).__init__()
        self.conv1 = nn.Conv2d(3,6,kernel_size=3,stride = 1, padding=0)

    def forward(self,x):
        x = self.conv1(x)
        return x

# 看看卷积后图像的输出是怎么样的
writer = SummaryWriter("logs")
a = Myclass()
step = 0
for imgs,targets in dataloader:
    output = a(imgs)
    writer.add_images("input",imgs,step, dataformats="NCHW")
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step, dataformats="NCHW")
    print(step)
    step += 1

writer.close()
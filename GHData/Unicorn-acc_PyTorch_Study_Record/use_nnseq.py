import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root = "./dataset", train=True,
                                       transform=torchvision.transforms.ToTensor(),download=False)
dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=True)


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule,self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024, 64)
        self.linear2 = nn.Linear(64, 10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

a = MyModule()

lr = 0.01

loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(a.parameters(), lr)
for epoch in range(20):
    runningloss = 0.0
    # 进行一轮的学习
    for imgs,target in dataloader:
        optim.zero_grad()
        l = loss(a(imgs),target)
        l.backward()
        optim.step()
        runningloss += l
    print(runningloss)

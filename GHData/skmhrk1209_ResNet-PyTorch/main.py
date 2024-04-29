import os
import argparse
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision import utils
from model import ResNet
from param import Param

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--checkpoint", type=str, default="")
args = parser.parse_args()

resnet = ResNet(
    conv_param=Param(in_channels=1, out_channels=32, kernel_size=7, stride=2),
    pool_param=Param(kernel_size=3, stride=2),
    residual_params=[
        Param(in_channels=32, out_channels=64, kernel_size=3, stride=1, blocks=1),
        Param(in_channels=64, out_channels=128, kernel_size=3, stride=2, blocks=1),
    ],
    num_classes=10,
    num_groups=32
)
print(resnet)

if args.checkpoint:
    resnet.load_state_dict(torch.load(args.checkpoint))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters())

dataset = datasets.MNIST(
    root="mnist",
    train=True,
    transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]),
    download=True,
)

data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=args.batch_size,
    shuffle=True
)

for epoch in range(args.num_epochs):
    for images, labels in data_loader:

        optimizer.zero_grad()

        logits = resnet(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        print(f"[{epoch}/{args.num_epochs}]: loss: {loss}")

    torch.save(resnet.state_dict(), f"model/epoch_{epoch}.pth")

dataset = datasets.MNIST(
    root="mnist",
    train=False,
    transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]),
    download=True,
)

data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=args.batch_size,
    shuffle=False
)

with torch.no_grad():

    correct = 0
    total = 0

    for images, labels in data_loader:

        logits = resnet(images)
        predictions = logits.argmax(1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    print(f"accuracy: {correct / total * 100.0}")

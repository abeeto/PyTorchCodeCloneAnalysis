from typing import Union

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import parallel

from timeit import timeit
import torchvision.transforms as transforms

import torchvision

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def run(model: Union[nn.Module, None] = None, batch_size: int = 1024, epoch: int = 1):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('data/', download=True, transform=transform, train=True), batch_size=batch_size,
        shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('data/', download=True, transform=transform, train=True), batch_size=batch_size,
        shuffle=True, num_workers=4)
    resnet = torchvision.models.resnet50(True).cuda()
    if model:
        resnet = model(resnet)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

    def train():
        for e in range(epoch):
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = resnet(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    print("Training time {:.2f}".format(timeit(lambda: train(), number=1)))
    with torch.no_grad():
        total, correct = .0, .0
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = resnet(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy {:.2f} %".format(100 * correct / total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help='')
    parser.add_argument('--epoch', default=1, type=int, help='')
    parser.add_argument('--batch-size', default=1024, type=int, help='')
    args = parser.parse_args()

    torch.manual_seed(0)
    model = None
    if args.model == 'future':
        model = parallel.DataParallelFuture
    elif args.model == 'threading':
        model = parallel.DataParallelThreading
    elif args.model == 'nn':
        model = nn.DataParallel
    elif args.model == 'single':
        model = None
    print('running model with {}'.format('no parallel' if not model else args.model))
    run(model=model, batch_size=args.batch_size, epoch=args.epoch)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data

from utils.eval import accuracy
from utils.model import alexnet
from utils.msic import AverageMeter


def train(model, dataloader, criterion, optimizer, epoch):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.cuda(0), targets.cuda(0, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print(f'Epoch: {epoch} Train: loss: {losses.avg} acc1: {top1.avg} acc5: {top5.avg}')


def test(model, dataloader, criterion, epoch):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(0), targets.cuda(0, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
        print(f'Epoch: {epoch} Test : loss: {losses.avg} acc1: {top1.avg} acc5: {top5.avg}')

def main():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 定义模型，数据集，数据加载器
    model = alexnet(num_classes=100).cuda(0)
    model = nn.DataParallel(model, device_ids=[0,1])
    train_dataset = datasets.CIFAR100(root='dataset', train=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='dataset', train=False, transform=transform_test)
    train_dataloader = data.DataLoader(train_dataset, 
                                    batch_size=64, sampler=data.RandomSampler(train_dataset), drop_last=True)
    test_dataloader = data.DataLoader(test_dataset, 
                                    batch_size=64, sampler=data.SequentialSampler(test_dataset), drop_last=True)
    # 损失函数，优化器, 学习率调度
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
    # 训练
    for epoch in range(1, 51):
        train(model, train_dataloader, criterion, optimizer, epoch)
        test(model, test_dataloader, criterion, epoch)
        lr_scheduler.step()
    


if __name__ == '__main__':
    main()

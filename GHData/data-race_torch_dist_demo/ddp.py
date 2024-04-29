import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data

from utils.eval import accuracy
from utils.model import alexnet
from utils.msic import AverageMeter


def train(model, dataloader, criterion, optimizer, epoch, device):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.cuda(device), targets.cuda(device, non_blocking=True)
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


def test(model, dataloader, criterion, epoch, device):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(device), targets.cuda(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
        print(f'Epoch: {epoch} Test : loss: {losses.avg} acc1: {top1.avg} acc5: {top5.avg}')

def main(**kwargs):
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
    # 准备工作
    import torch.distributed as dist
    device = kwargs['device']
    local_rank = kwargs['local_rank']
    world_size = kwargs['world_size']
    init_method = kwargs['init_method'] # tcp://localhost:9999
    dist.init_process_group(backend='nccl', world_size=world_size, rank=local_rank, init_method=init_method)
    # 定义模型，数据集，数据加载器
    model = alexnet(num_classes=100).cuda(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    train_dataset = datasets.CIFAR100(root='dataset', train=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='dataset', train=False, transform=transform_test)
    train_dataloader = data.DataLoader(train_dataset, 
                                    batch_size=64, sampler=data.DistributedSampler(train_dataset), drop_last=True)
    test_dataloader = data.DataLoader(test_dataset, 
                                    batch_size=64, sampler=data.SequentialSampler(test_dataset), drop_last=True)
    # 损失函数，优化器, 学习率调度
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
    # 训练
    for epoch in range(1, 51):
        train(model, train_dataloader, criterion, optimizer, epoch, device)
        test(model, test_dataloader, criterion, epoch, device)
        lr_scheduler.step()
    


if __name__ == '__main__':
    from multiprocessing import Process
    p_0 = Process(target=main, kwargs={'device':0, 'local_rank':0, 'world_size':2, 'init_method': 'tcp://localhost:9999'})
    p_1 = Process(target=main, kwargs={'device':1, 'local_rank':1, 'world_size':2, 'init_method': 'tcp://localhost:9999'})
    p_0.start()
    p_1.start()
    p_0.join()
    p_1.join()

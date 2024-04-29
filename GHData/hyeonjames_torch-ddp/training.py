import os
from math import ceil
from typing import Callable, Set, Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.multiprocessing import Process

import partition


def setup(rank: int, world_size: int, fn: Callable, args: Set[Any] = (), backend: str = 'nccl') -> None:
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.manual_seed(42)
    torch.cuda.set_device(torch.device(f'cuda:{rank}'))
    fn(torch.device(f'cuda:{rank}'), *args)


def run(device: torch.device, epochs: int = 1, batch_size: int = 4096) -> None:
    rank = dist.get_rank()
    model = torchvision.models.resnet50().to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    ])
    dataset = torchvision.datasets.CIFAR10('/data/private/datasets', train=True, transform=transform, download=True)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # partition.partiton_dataset is same as the one in pytorch tutorial
    # https://pytorch.org/tutorials/intermediate/dist_tuto.html#distributed-training
    train_set, bsz = partition.partition_dataset(dataset, batch_size)
    num_batches = ceil(len(train_set.dataset)) / float(bsz)

    for epoch in range(epochs):
        epoch_loss = .0
        for data, target in train_set:
            optimizer.zero_grad()
            data = data.to(device)  # all of processes are locked here at the second iteration
            target = target.to(device)
            outputs = model(data)
            loss = F.cross_entropy(outputs, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print(f'Rank {rank}, epoch {epoch} : {epoch_loss / num_batches}')


def average_gradients(model: nn.Module) -> None:
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


if __name__ == '__main__':
    world_size = 4
    epoch = 1
    batch_size = 4096
    processes = []
    for rank in range(world_size):
        p = Process(target=setup, args=(rank, world_size, run, (epoch, batch_size), 'nccl'))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

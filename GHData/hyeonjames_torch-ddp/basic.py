import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torchvision
import torchvision.transforms as tf
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank: int, world_size: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(42)


def cleanup():
    dist.destroy_process_group()


def run_process(rank: int, world_size: int) -> None:
    setup(rank, world_size)

    n = torch.cuda.device_count() // world_size
    device_ids = list(range(rank * n, (rank + 1) * n))

    resnet = torchvision.models.resnet50(True)

    transform = tf.Compose([
        tf.ToTensor(),
        tf.Normalize([.5, .5, .5], [.5, .5, .5])
    ])

    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('/data/private/datasets', train=True, transform=transform, download=True),
        batch_size=4096,
        shuffle=True,
        num_workers=1, pin_memory=True)
    model = resnet.to(device_ids[0])
    ddp_model = DDP(model, device_ids=device_ids)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    for data in trainloader:
        inputs, labels = data
        optimizer.zero_grad()
        inputs = inputs.to(device_ids[0])
        labels = labels.to(device_ids[0])
        outputs = ddp_model(inputs)
        loss_fn(outputs, labels).backward()

        optimizer.step()
    cleanup()


if __name__ == '__main__':
    mp.spawn(run_process, args=(2,), nprocs=2, join=True)

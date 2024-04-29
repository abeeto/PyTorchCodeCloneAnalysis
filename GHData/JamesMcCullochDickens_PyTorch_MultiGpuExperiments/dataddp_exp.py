import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision.models import resnet50
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from sys import platform
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler


def is_windows():
    return platform == "win32"


def spawn_processes(fn, world_size):
    mp.spawn(fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # TODO was having some issues with nccl hanging with larger batch sizes, to investigate
    #dist.init_process_group("gloo" if is_windows() else "nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class RandomImagesDataloader(Dataset):
    def __init__(self, num_images=1000, height=600, width=600, num_channels=3):
        self.num_images = num_images
        self.dataset = torch.randn(num_images, num_channels, height, width)
        self.len = num_images

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.len


def train(rank, world_size):
    setup(rank, world_size)
    model = resnet50().to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)
    batch_sizes = [8, 16]
    num_epochs = 5
    # warmup iterations
    for i in range(10):
        sample_input = torch.rand(10, 3, 600, 600).to(rank)
        _ = model(sample_input)
    for batch_size in batch_sizes:
        dataset = RandomImagesDataloader()
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
        dl = DataLoader(sampler=sampler, dataset=dataset,
                        batch_size=batch_size, shuffle=False,
                        num_workers=0, drop_last=True)
        optimizer = optim.SGD(params=model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        total_time = 0.0
        for epoch_num in range(num_epochs):
            for batch_num, batch in enumerate(dl):
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start_event.record()
                targets = torch.randint(size=(batch_size,), low=0, high=1000).long().to(rank)
                batch = batch.to(rank)
                output = model(batch)
                loss = loss_fn(output, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                end_event.record()
                torch.cuda.synchronize()
                total_time += start_event.elapsed_time(end_event)
        if rank == 0:
            print(f"The estimated training time for {world_size} gpu/s at batch size "
                  f"{batch_size} is {round(total_time/1000.0, 3)} seconds")
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    spawn_processes(train, world_size)

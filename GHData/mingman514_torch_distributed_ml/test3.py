import os
import socket
import torch
import torch.distributed as dist

def run(rank, size, hostname):
    """ Distributed function to be implemented later. """
    print(f'This is {rank} of {size} in {hostname}')
    tensor = torch.zeros(1)

    if rank == 0:
        tensor += 1
        dist.send(tensor=tensor, dst=1)

    else:
        dist.recv(tensor=tensor, src=0)

    print('Rank ', rank, ' has data ', tensor[0])

def init_processes(rank, size, hostname, fn, backend='mpi'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, hostname)


if __name__ == "__main__":
    world_size = int(os.environ['MV2_COMM_WORLD_SIZE'])
    world_rank = int(os.environ['MV2_COMM_WORLD_RANK'])
    hostname = socket.gethostname()

    print(f'world_size: {world_size}, world_rank: {world_rank}, hostname: {hostname}')
    init_processes(world_rank, world_size, hostname, run, backend='mpi')

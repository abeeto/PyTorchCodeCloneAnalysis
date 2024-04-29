#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def main(length):
    """Set up an array of specified length and gather it back to the root process."""
    rank = dist.get_rank()
    comm_size = dist.get_world_size()

    print(f'Starting rank {rank} of {comm_size}')

    x = torch.ones(length) * rank # Default type is float, which is a good choice.

    buf = [torch.empty(length) for i in range(comm_size)]

    dist.all_gather(buf, x)      # Synchronous collective: all processes block until complete.

    if rank==0:
        rslt = torch.stack(buf)
        print(f'rank: {rank}:\n{rslt}')
    else:
        print(f'rank: {rank}:  done.\n')


def init_proc(rank, size, run, backend, mainarg):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '24601'
    dist.init_process_group(backend, rank=rank, world_size=size)
    run(mainarg)


if __name__ == '__main__':

    size = 2
    procs = []
    length = 1024

    for rank in range(size):
        p = Process(target=init_proc, args=(rank, size, main, 'gloo', length))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()



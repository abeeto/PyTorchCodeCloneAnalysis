#!/usr/bin/env python
import torch
import torch.distributed as dist

def main(length):
    """Set up an array of specified length and gather it back to the root process."""
    rank = dist.get_rank()
    x = torch.ones(length) * rank # Default type is float, which is a good choice.

    if rank==0:
        nproc = dist.get_world_size()
        buf = [torch.empty(length) for i in range(nproc)]
    else:
        buf = None

    dist.gather(x, buf, 0)      # Synchronous collective: all processes block until complete.

    if rank==0:
        rslt = torch.stack(buf)
        print(f'rank: {rank}:\n{rslt}')
    else:
        print(f'rank: {rank}:  done.\n')

if __name__ == '__main__':
    dist.init_process_group('mpi')
    main(1024)

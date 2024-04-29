# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
import threading
import time
import torch
import torch.distributed as dist

NUM_TRIALS = 20

def all_reduce_helper(tensor, group, multiplier, num_iterations):
    dist.barrier()
    start_time = time.time()
    for i in range(num_iterations):
        dist.all_reduce(tensor=tensor, group=group)
    dist.barrier()
    size = tensor.size()[0]
    bandwidth = (size * 4. * NUM_TRIALS * multiplier) / ((time.time() - start_time) * 10**6)
    print("[%d] Bandwidth for tensor size %s: %.2f MB/s" % (dist.get_rank(), size, bandwidth))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test lightweight communication library')
    parser.add_argument("--backend", type=str, default='nccl',
                        help="Backend")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="Local rank of current worker")

    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)

    dist.init_process_group(args.backend, 
                            init_method='file:///home/hamt5821/sharedfile', 
                            rank=os.getenv('SLURM_PROCID'), 
                            world_size=os.getenv('SLURM_NTASKS'))

    tensor_sizes = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
    world_size = dist.get_world_size()
    groups = []
    for tag in range(len(tensor_sizes)):
        group = dist.new_group(list(range(world_size)))
        groups.append(group)

    multiplier = (2. * (world_size-1)) / world_size
    for tag, tensor_size in enumerate(tensor_sizes):
        group = groups[tag]
        tensor = torch.tensor(range(tensor_size), dtype=torch.float32).cuda(args.local_rank)
        all_reduce_helper(tensor, group, multiplier, NUM_TRIALS)

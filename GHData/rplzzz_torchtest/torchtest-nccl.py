#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import argparse

def main(length, dev):
    """Set up an array of specified length and gather it back to the root process."""
    rank = dist.get_rank()
    comm_size = dist.get_world_size()

    ofname = f'log-{rank}.txt'

    with open(ofname, 'w') as of:
        of.write(f'Starting rank {rank} of {comm_size}\n')
        of.write(f'Device = {dev}\n')
        of.flush()

        ## Default type is float, which is a good choice.
        x = torch.ones(length) * rank 
        x = x.cuda() # Weirdly, trying to do this conversion all in one line caused the program to hang.
        
        buf = [torch.empty(length).to(torch.device('cuda')) for i in range(comm_size)]
        
        dist.all_gather(buf, x)

        if rank==0:
            rslt = torch.stack(buf)
            of.write(f'rank: {rank}:\n{rslt}')
            of.write('\n')
        else:
            of.write(f'rank: {rank}:  done.\n')
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    args=parser.parse_args()

    lrank = args.local_rank
    print(f'localrank: {lrank}   host: {os.uname()[1]}')
    torch.cuda.set_device(lrank)

    dist.init_process_group('nccl', 'env://')
    main(1024, lrank)


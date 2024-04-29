import os
import tempfile
import dnnlib

import torch

def subprocess_fn(rank, c, temp_dir):
    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    if rank == 0:
        print('< original tensor >')
    torch.distributed.barrier() #Synchronizes all processes.
    """
    If we don't insert barrier, print log doesn't sync.
    e.g.
    < original tensor >
    rank0: tensor([0, 0, 0, 0, 1, 0, 0, 1], device='cuda:0')
    < gather and sum >
    rank1: tensor([1, 0, 0, 0, 1, 0, 0, 1], device='cuda:1')
    rank0: tensor([1, 0, 0, 0, 2, 0, 0, 2], device='cuda:0')
    """

    device = torch.device('cuda', rank)
    tensor = torch.randint(low=0, high=2, size=(8, )).to(device)
    print(f'rank{rank}: {tensor}')

    if c.num_gpus > 1:
        gatherd = [torch.zeros_like(tensor) for i in range(c.num_gpus) ]
        torch.distributed.all_gather(gatherd, tensor)
    
        if rank == 0:
            tensor = torch.zeros_like(tensor)
            for t in gatherd:
                tensor = tensor + t
        
    if rank == 0:
        print('< gather and sum >')
        print(f'rank{rank}: {tensor}')

    if c.num_gpus > 1:
        torch.distributed.broadcast(tensor, src=0)

    if rank == 0:
        print('< broadcast to all gpus >')

    print(f'rank{rank}: {tensor}')

if __name__ == '__main__':
    c = dnnlib.EasyDict(num_gpus=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

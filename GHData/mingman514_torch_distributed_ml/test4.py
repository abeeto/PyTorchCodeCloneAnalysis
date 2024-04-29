import torch
import torch.distributed as dist

def main(rank, world):
    print(f'this is {rank} of {world}')
    if rank == 0:
        x = torch.tensor([1., -1.])
        #dist.send(x, dst=1)
        print('Rank-0 has sent the tensor to Rank-1')
        print(x)

    else:
        z = torch.tensor([0., 0.])
        #dist.recv(z, src=0)
        print('Rank-1 has recieved the tensor from Rank-0')
        print(z)

if __name__ == '__main__':

    
    dist.init_process_group(backend='mpi')
    print(f'rank: {dist.get_rank()}, world: {dist.get_world_size()}')
    main(dist.get_rank(), dist.get_world_size())

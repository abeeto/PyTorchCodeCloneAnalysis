import os
import socket

import torch
import torch.distributed as dist
# import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp(global_rank, local_rank, world_size, port_num):
    current_dir = os.getcwd()
    with open(current_dir + "/hostfile") as f:
        host = f.readlines()
    host[0] = host[0].rstrip("\n")
    # os.environ["NCCL_SOCKET_IFNAME"] = f"ens85f0,ib{local_rank // 2 * 2}"
    # os.environ["NCCL_SOCKET_IFNAME"] = f"bond0"
    dist_url = "tcp://" + host[0] + ":" + str(port_num)
    print(dist_url)
    # initialize the process group
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        rank=global_rank,
        world_size=world_size
    )
    print("tcp connected")


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def cleanup():
    dist.destroy_process_group()


# def main(local_rank, world_size, node_rank, local_size, port_num):
def main():
    """ Main """
    # Setup Distributed Training
    # gpu_rank: 0~4 in ABCI, i.e. intra gpu rank
    # world size: total process num
    # print(f"node rank: {node_rank}")
    # rank = local_size * node_rank + local_rank  # global gpu rank
    hostname = socket.gethostname()
    #if rank == 0:
        # print("num_gpu:{}".format(local_size))
    # print(f"global rank {rank} of {world_size} / local {local_rank} on node {hostname} (n {node_rank}) ")
    # set up communication setting between nodes
    # setup_ddp(global_rank=os.environ["GLOBAL_RANK"],
    #           local_rank=local_rank,
    #           world_size=world_size,
    #           port_num=port_num)
    dist_url = "tcp://" + os.environ["MASTER_ADDR"] + ":" + os.environ["MASTER_PORT"]
    print(dist_url)
    # initialize the process group
    global_rank = int(os.environ["GLOBAL_RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    node_rank = int(os.environ["NODE_RANK"])
    print(f"global rank {global_rank} of {world_size} / local {local_rank} on node {hostname} (node {node_rank}) ")
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        rank=global_rank,
        world_size=world_size
    )

    # ------------- training logic from here -----------
    model = ToyModel().cuda()
    ddp_model = DDP(model, device_ids=[0])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).cuda()
    loss_fn(outputs, labels).backward()
    optimizer.step()
    print(f"rank {global_rank} step is done")
    # cleanup()


if __name__ == '__main__':
    # ddp setup
    #node_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])  # Process number in MPI
    #size = int(os.environ["OMPI_COMM_WORLD_SIZE"])  # The all size of process
    #print("node rank:{}".format(node_rank))
    #print("size of process:{}".format(size))
    # cnt_gpus = torch.cuda.device_count()  # gpu num per node
    # gpu = 1
    #world_size = cnt_gpus * size  # total gpu num
    #print(f"target world size {world_size}")

    #port_num = 50000
    # mp.spawn(main, nprocs=cnt_gpus, args=(world_size, node_rank, cnt_gpus, port_num))
    main()

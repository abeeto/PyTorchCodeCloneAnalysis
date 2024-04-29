from client import client
from server import server
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from model import *
import os
import logging
import argparse
from data.iris_data_generator import *
from data.relational_table_preprocessor import relational_table_preprocess_dl

def init_env():
    print("Initialize Meetup Spot")
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ["MASTER_PORT"] = "5689"

def example(rank,world_size,args):
    init_env()
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)
    if rank == 0:
        Server = server(LogisticRegression(4,3),
                        rank,
                        world_size,
                        args)

        Server.update_clients()

        for iter in range(args.iterations):
            Server.train()
            Server.update_clients()
            Server.evaluate()
        rpc.shutdown()
    else:
        print("Client")
        rpc.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FedAVG Initialization')
    parser.add_argument('--world_size',type=int,default=3,help='The world size which is equal to 1 server + (world size - 1) clients')
    parser.add_argument('--epochs',type=int,default=3,help='The number of epochs to run on the client training each iteration')
    parser.add_argument('--iterations',type=int,default=1000,help='The number of iterations to communication between clients and server')
    parser.add_argument('--batch_size',type=int,default=16,help='The batch size during the epoch training')
    parser.add_argument('--partition_alpha',type=float,default=0.5,help='Number to describe the uniformity during sampling (heterogenous data generation for LDA)')

    args = parser.parse_args()

    args.client_num_in_total = args.world_size - 1

    load_iris(args)

    world_size = args.world_size
    mp.spawn(example,
             args=(world_size,args),
             nprocs=world_size,
             join=True)
import os
import socket
import sys

from mpi4py import MPI


def get_address():

    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        st.connect(('10.255.255.255', 1))
        IP = st.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
        raise RuntimeError("can't determine routable IP")
    finally:
        st.close()
    return IP


def main():
    comm = MPI.COMM_WORLD
    global_rank = comm.Get_rank()
    world_size = comm.Get_size()
    cnt_gpus = 8
    # print("setting up ")

    os.environ["MASTER_PORT"] = "31415"
    os.environ["NODE_RANK"] = str(global_rank // cnt_gpus)
    local_rank = global_rank % cnt_gpus
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["GLOBAL_RANK"] = str(global_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    # os.environ["CNT_NODES"] = str(glosize)
    # PL_TORCH_DISTRIBUTED_BACKEND=gloo
    master_addr = get_address()
    master_addr = comm.bcast(master_addr, root=0)
    os.environ["MASTER_ADDR"] = master_addr
    # TODO: check if port is availalbe at master rank

    os.system(" ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()

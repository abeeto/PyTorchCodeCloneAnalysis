import argparse
import torch
import torch.distributed


PORT = 23456


def init_process_group(rank, world_size, ip, port=PORT):
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"tcp://{ip}:{port}",
        rank=rank,
        world_size=world_size,
    )

    print("Torch distributed initialized:", torch.distributed.is_initialized())


def print_available_features():
    print("CUDA available:", torch.cuda.is_available())
    print("Distributed available:", torch.distributed.is_available())
    print("NCCL (distributed GPU) available:", torch.distributed.is_nccl_available())


def main():
    parser = argparse.ArgumentParser(
        description="Run a small distributed computation, possibly using CUDA"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="The program's rank. A number from 0 to --world-size - 1. The 0th process is the leader.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        required=True,
        help="The number of processes participating in the computation.",
    )
    parser.add_argument(
        "--ip", type=str, required=True, help="The rank 0 program's IP address."
    )

    args = parser.parse_args()

    print_available_features()

    init_process_group(rank=args.rank, world_size=args.world_size, ip=args.ip)


if __name__ == "__main__":
    main()

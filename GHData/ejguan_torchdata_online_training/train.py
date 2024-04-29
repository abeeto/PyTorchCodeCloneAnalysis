import torch.distributed as dist

from torchdata.dataloader2 import DataLoader2
from torchdata.datapipes.iter import IterableWrapper

from reading_service import DistributedReadingService


def main():
    dist.init_process_group("nccl")

    #  ws = dist.get_world_size()
    rank = dist.get_rank()

    dp = IterableWrapper(list(range(9))).shuffle().sharding_filter()

    rs = DistributedReadingService()

    dl = DataLoader2(dp, reading_service=rs)

    for epoch in range(2):
        print(f"Rank: {rank}, Epoch: {epoch+1}")
        for idx, d in enumerate(dl):
            print(f"Rank: {rank}, ID: {idx}, Data: {d}")
            dist.barrier()


if __name__ == "__main__":
    main()

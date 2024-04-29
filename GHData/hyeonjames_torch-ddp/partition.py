from random import Random
from typing import Iterable, Union, List, Any

import torch.distributed as dist
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class Partition(Dataset):
    def __init__(self, data: Dataset, index: List[int]) -> None:
        self.index = index
        self.data = data

    def __getitem__(self, index: Any) -> Tensor:
        return self.data[self.index[index]]

    def __len__(self) -> int:
        return len(self.index)


class Partitioner(object):
    def __init__(self, data: Dataset, frags: Iterable[int], seed=1234) -> None:
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_size = len(data)
        indexes = list(range(data_size))
        rng.shuffle(indexes)
        for frag in frags:
            size = int(data_size * frag)
            self.partitions.append(indexes[:size])
            indexes = indexes[size:]

    def use(self, partition: int) -> Partition:
        return Partition(self.data, self.partitions[partition])


def partition_dataset(data: Dataset, batch_size: int, partition_sizes: Union[None, Iterable[int]] = None) -> DataLoader:
    world_size = dist.get_world_size()
    bsz = batch_size // world_size
    if not partition_sizes:
        partition_sizes = [1 / world_size] * world_size
    partitioner = Partitioner(data, partition_sizes)
    partition = partitioner.use(dist.get_rank())

    return DataLoader(partition, batch_size=bsz, shuffle=True), bsz

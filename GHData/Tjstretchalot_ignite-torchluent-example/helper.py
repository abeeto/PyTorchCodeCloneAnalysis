"""Helper functions"""
import torch
import numpy as np

def create_rand_subset(dataset: torch.utils.data.Dataset, num: int):
    if num >= len(dataset):
        return dataset

    inds = np.random.choice(len(dataset), (num,), False)
    inds = torch.from_numpy(inds).int()
    return torch.utils.data.Subset(dataset, inds)

def create_loader_for_subset(dataset: torch.utils.data.Dataset, num: int, **kwargs):
    subs = create_rand_subset(dataset, num)
    return torch.utils.data.DataLoader(subs, **kwargs)

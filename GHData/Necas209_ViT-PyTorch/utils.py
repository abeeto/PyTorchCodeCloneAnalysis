from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import scipy.io as sio
import torch
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader


@dataclass
class AverageMeter:
    """Computes and stores the average and current value"""
    val: float | torch.Tensor = 0.
    avg: float | torch.Tensor = 0.
    sum: float | torch.Tensor = 0.
    count: float | torch.Tensor = 0.

    def reset(self) -> None:
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val: float | torch.Tensor, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_data(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Loads data, either from .mat file or saved dictionary, and returns tuple of DataLoaders"""
    if args.data is False:
        # Load data from .mat file and clean it
        mat = sio.loadmat('Hyper_2012_TF_sugar.mat')
        x: np.ndarray = mat['MeanROffPedNorm']
        y: np.ndarray = mat['sugar']
        x = x.transpose()[:, 10:-6].reshape(240, 1, 32, 32)
        t_x = torch.Tensor(x)
        t_y = torch.Tensor(y)
        # Shuffle tensors' rows
        indices = torch.randperm(t_x.size()[0])
        t_x = t_x[indices]
        t_y = t_y[indices]
        # Split dataset into train, validation and test sets
        dataset = TensorDataset(t_x, t_y)
        ds_size = len(dataset)
        test_size = int(0.1 * ds_size)
        train_size = ds_size - test_size
        ds, test_ds = data.random_split(dataset, [train_size, test_size])
        ds_size = len(ds)
        val_size = int(0.1 * ds_size)
        train_size = ds_size - val_size
        train_ds, val_ds = data.random_split(ds, [train_size, val_size])
        # Create data loaders
        kwargs = {'num_workers': 4, 'pin_memory': True}
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, **kwargs)
        # Save data loaders to file
        if args.resume is False:
            d = {'train_loader': train_loader,
                 'val_loader': val_loader,
                 'test_loader': test_loader}
            torch.save(d, f'./runs/{args.name}/dataloader_dict.pt')
    else:
        dl_path = f'./runs/{args.name}/dataloader_dict.pt'
        print(f'Loading data from {dl_path}')
        d = torch.load(dl_path)
        train_loader = d['train_loader']
        val_loader = d['val_loader']
        test_loader = d['test_loader']
    return train_loader, val_loader, test_loader

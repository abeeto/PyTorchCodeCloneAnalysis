import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
import typing


class MNISTDataset:
    def __init__(self, path_to_data: str, batch_size: int=128, flatten: bool=False):
        self.path = path_to_data
        self.bs = batch_size
        if not flatten:
            self.shape = (-1, 1, 28, 28)
        else:
            self.shape = (-1, 784)

    def get_iterators(self) -> typing.Tuple[DataLoader, DataLoader]:

        with open(self.path, 'rb') as f:
            ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
        x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

        train_ds = TensorDataset(x_train.view(self.shape), y_train)
        valid_ds = TensorDataset(x_valid.view(self.shape), y_valid)

        train_dl = DataLoader(train_ds, batch_size=self.bs)
        valid_dl = DataLoader(valid_ds, batch_size=self.bs)

        return train_dl, valid_dl

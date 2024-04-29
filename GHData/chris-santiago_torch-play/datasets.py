from dataclasses import dataclass, astuple
from numpy import ndarray
from sklearn.preprocessing import LabelBinarizer


@dataclass
class Dataset:
    train_x: ndarray
    train_y: ndarray
    valid_x: ndarray
    valid_y: ndarray

    def __iter__(self):
        return iter(astuple(self))


class Mnist:
    def __init__(self, dataset: Dataset):
        self.data = dataset

    def clean(self, binarize=True):
        if binarize:
            lb = LabelBinarizer()
            return Dataset(
                self.data.train_x.reshape(-1, 784) / 255.,
                lb.fit_transform(self.data.train_y),
                self.data.valid_x.reshape(-1, 784) / 255.,
                lb.transform(self.data.valid_y)
            )
        return Dataset(
            self.data.train_x.reshape(-1, 784) / 255.,
            self.data.train_y,
            self.data.valid_x.reshape(-1, 784) / 255.,
            self.data.valid_y
        )

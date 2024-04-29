import datetime
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch.utils.data import Sampler

import config


class SubsetSequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, start_idx=0):
        super(SubsetSequentialSampler, self).__init__(data_source)
        assert 0 <= start_idx < len(data_source)
        self.data_source = data_source
        self.start_idx = start_idx

    def __iter__(self):
        return iter(range(self.start_idx, len(self.data_source)))

    def __len__(self):
        return len(self.data_source) - self.start_idx


def min_max_scale(x):
    denom = x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True)
    assert denom.sum() != 0.0
    x = (x - x.min(axis=0, keepdims=True)) / denom
    return x


def train_test_split(n_samples: int, proportion: float = 0.8):
    training_set_idx = int(n_samples * proportion)
    test_set_idx = int(n_samples - training_set_idx)
    return training_set_idx, test_set_idx


def standardize(x):
    denom = x.std(axis=0, keepdims=True) + 1e-6
    assert denom.sum() != 0.0
    x = (x - x.mean(axis=0, keepdims=True)) / denom
    return x


def _build_dataset_microgrid(window=4, period=24):
    df = pd.read_csv(
        config.ORIGINAL_PATH,
        sep=";|,",
        parse_dates=True,
        index_col="DateTime",
        engine="python",
    )
    # 5min to 1hr
    df[df < 0] = 0
    df = df.resample("1h").apply(np.mean)

    assert not df.isnan().sum()
    assert df.index.is_monotonic_increasing

    idx = df.index.to_list()
    x = df.values[:, :1]

    t_0 = period * 2 + window // 2 + 1
    xs = []
    ys = []
    x_dates = []
    y_dates = []

    for t in range(t_0, len(idx) - 2):
        # for t in idx:
        # at time t  off by one, last point takes value one hr e
        dm1_slice = slice(t - period - window // 2, t - period + window // 2 + 1)
        dm2_slice = slice(
            t - period * 2 - window // 2, t - period * 2 + window // 2 + 1
        )
        recent_past_slice = slice(t - window, t + 1)
        y_slice = slice(t + 1, t + 2)
        idxs = idx[recent_past_slice] + idx[dm1_slice] + idx[dm2_slice]
        dm1 = x[dm1_slice]
        dm2 = x[dm2_slice]
        recent_past = x[recent_past_slice]
        _x = np.concatenate([recent_past, dm1, dm2])
        assert len(_x) == len(idxs)
        xs.append(_x)
        x_dates.append(idxs)

        ys.append(x[y_slice][0])
        y_date = idx[y_slice][0]
        assert y_date - max(idxs) == datetime.timedelta(hours=1)
        y_dates.append(y_date)

    xs = np.stack(xs)
    ys = np.stack(ys)
    print(xs.shape)
    assert len(xs) == len(ys)
    import pickle

    with open("data/dataset.pkl", "wb") as f:
        pickle.dump((xs, ys), f)

    # want all values up to time -window / 2
    # the window +/- window/2 of the current day and the day before


def get_data_loaders(batch_size=32, proportion=0.8, start_idx=0, test_batch_size=1):
    with open("data/dataset.pkl", "rb") as f:
        x, y = pickle.load(f)
    x = torch.FloatTensor(x).transpose(2, 1)
    y = torch.FloatTensor(y)
    tr_idx, ts_idx = train_test_split(len(x), proportion=proportion)
    tr_set = data.TensorDataset(x[:tr_idx], y[:tr_idx])
    ts_set = data.TensorDataset(x[tr_idx:], y[tr_idx:])
    tr_set = data.DataLoader(tr_set, batch_size=batch_size, shuffle=True)
    ts_set = data.DataLoader(
        ts_set,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=True,
        sampler=SubsetSequentialSampler(ts_set, start_idx=start_idx),
    )
    shape = x.shape
    return (tr_set, ts_set, shape)


if __name__ == "__main__":
    _build_dataset_microgrid()

import matplotlib.pyplot as plt
import torch

from dataset import get_data_loaders
from utils.viz_utils import plot_output
from argparse import ArgumentParser
import models

import config


def _ar(batch, model):
    x, y = batch
    y = y[:-1]
    train_batch_size = x.shape[0]
    obs_idx = 4
    _x = x[0].unsqueeze(dim=0)
    ys = []
    for n in range(0, train_batch_size - 1):
        y_hat = model(_x)
        _x = x[n + 1].unsqueeze(dim=0)
        _x[:, :, obs_idx] = y_hat
        ys.append(y_hat)
    y_hat = torch.cat(ys)
    assert len(y) == len(y_hat)
    return y, y_hat


@torch.no_grad()
def ar(dataset, model):
    model.eval()
    x, y = iter(dataset).__next__()
    out = _ar((x, y), model)
    return out


def main(args, seed):
    torch.random.manual_seed(seed)

    torch.random.manual_seed(seed)
    train_loader, val_loader, shape = get_data_loaders(
        config.Training.batch_size,
        start_idx=args.start_idx,
        test_batch_size=args.horizon,
    )
    n, d, t = shape
    model = models.ConvNet(d, seq_len=t)
    if args.ckpt is not None:
        state_dict = torch.load(args.ckpt)
        model.load_state_dict(state_dict)

    out = ar(val_loader, model)
    plot_output(*out)
    plt.show()
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--start-idx", type=int, default=0, help="start idx",
    )
    parser.add_argument(
        "--horizon", type=int, default=48, help="forecast horizon",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="model path",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="general seed",
    )
    parser.add_argument(
        "--verbose", type=int, default=0, help="...",
    )

    args = parser.parse_args()

    main(args, seed=args.seed)

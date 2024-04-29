import datetime
import pickle
from typing import Literal, Optional, Any, cast
from tqdm.auto import tqdm, trange  # type: ignore
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchinfo import summary

from torch_numbers.utils import count_params, mask_images, fill_mask, display_imgs
from torch_numbers.data import get_dataset, DATASET_OPTIONS
from torch_numbers.base_models import Encoder


def main(
        n_epochs: int = 50,
        batch_size: int = 64,
        encoding_dim: int = 30,
        print_model_summary: bool = False,
        pbar_smoothing: float = 0.03,
        save_encoder: Optional[str] = 'pretrained_encoder.pt',
        save_hist: Optional[str] = 'hist.pkl',
):
    train_loader, encoder, recreate_mask, fill_size, opt = prepare(
        batch_size=batch_size,
        encoding_dim=encoding_dim,
        print_model_summary=print_model_summary,
    )

    hist = train(
        train_loader, encoder, recreate_mask, fill_size, opt,
        n_epochs=n_epochs, pbar_smoothing=pbar_smoothing,
        ipy_display=False, delete_file=False,
    )
    if save_hist is not None:
        with open(save_hist, 'wb') as f:
            pickle.dump(hist, f)

    if save_encoder is not None:
        torch.save(encoder, save_encoder)
    return encoder


class MaskRecreator(nn.Module):

    def __init__(
            self,
            relu_negative_slope: float = 0.1,
            norm: Literal['batch', 'instance'] = 'instance',
            in_dim: int = 256,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Unflatten(-1, (in_dim, 1, 1)),
            nn.ConvTranspose2d(in_dim, 64, kernel_size=(2, 2)),
            nn.LeakyReLU(relu_negative_slope, inplace=True),
            {'batch': nn.BatchNorm2d(64), 'instance': nn.InstanceNorm2d(64)}[norm],
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.LeakyReLU(relu_negative_slope, inplace=True),
            {'batch': nn.BatchNorm2d(32), 'instance': nn.InstanceNorm2d(32)}[norm],
            nn.ConvTranspose2d(32, 8, kernel_size=(4, 4), stride=(2, 2)),
            nn.LeakyReLU(relu_negative_slope, inplace=True),
            {'batch': nn.BatchNorm2d(8), 'instance': nn.InstanceNorm2d(8)}[norm],
            nn.Conv2d(8, 1, kernel_size=(1, 1)),
            nn.Sigmoid(),
        )

        self.example_input_array = torch.rand((2, in_dim))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.model(x)


def train_batch(encoder: nn.Module, recreate_mask: nn.Module, opt: torch.optim.Optimizer,
                imgs: Tensor, fill_size: tuple[int, int], log: Optional[dict[str, Any]] = None):
    opt.zero_grad()

    imgs_masked, masked, _ = mask_images(imgs, fill_size, fill=0.5)
    z = encoder(imgs_masked)
    recreated = recreate_mask(z)
    loss = F.binary_cross_entropy(recreated, masked)

    loss.backward()
    opt.step()

    if log is None:
        log = dict()
    log['time'] = datetime.datetime.now()
    log['batch_size'] = len(imgs)
    log['loss'] = loss.item()

    return log


def prepare(
        dataset: DATASET_OPTIONS = 'MNIST',
        batch_size: int = 64,
        encoding_dim: int = 30,
        print_model_summary: bool = False,
):
    # data
    train_ds = get_dataset(dataset, normalize='sigmoid')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # models
    encoder = Encoder(out_dim=encoding_dim)
    if print_model_summary:
        print(summary(encoder, input_data=encoder.example_input_array,
                      col_names=["input_size", "output_size", "num_params"]))
    else:
        print(f"Encoder has {count_params(encoder, trainable=True):,d} trainable parameters")

    recreate_mask = MaskRecreator(in_dim=encoding_dim)
    if print_model_summary:
        print(summary(recreate_mask, input_data=recreate_mask.example_input_array,
                      col_names=["input_size", "output_size", "num_params"]))
    else:
        print(f"MaskRecreator has {count_params(recreate_mask, trainable=True):,d} trainable parameters")
    fill_size = cast(tuple[int, int], tuple(recreate_mask(recreate_mask.example_input_array).shape[-2:]))

    # optimizer
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(recreate_mask.parameters()),
        lr=0.001, betas=(0.9, 0.999),
    )

    return train_loader, encoder, recreate_mask, fill_size, opt


def train(
        train_loader: DataLoader,
        encoder: nn.Module,
        recreate_mask: nn.Module,
        fill_size: tuple[int, int],
        opt: torch.optim.Optimizer,
        n_epochs: int = 5,
        pbar_smoothing: float = 0.03,
        ipy_display: bool = True,
        delete_file: bool = True,
) -> list[dict[str, Any]]:
    hist = []
    expw: dict[str, Any] = dict()
    n_examples: int = len(train_loader.sampler)  # type: ignore
    for epoch in trange(n_epochs):
        with tqdm(total=n_examples, smoothing=pbar_smoothing) as pbar:
            for batch_idx, (imgs, lbls) in enumerate(train_loader):
                log = {
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                }
                train_batch(encoder, recreate_mask, opt, imgs, fill_size, log=log)

                hist.append(log)

                for k, v in log.items():
                    if isinstance(v, (int, float)):
                        expw[k] = pbar_smoothing * v + (1 - pbar_smoothing) * expw.get(k, v)
                pbar.set_postfix_str("loss={loss:.4}".format(**expw))
                pbar.update(len(imgs))

        n_columns = 30
        for imgs, lbls in train_loader:
            imgs = imgs[:n_columns]
            break
        imgs_masked, masked, pos = mask_images(imgs, fill_size, fill=0.5)
        z = encoder(imgs_masked)
        recreated_masks = recreate_mask(z)
        recreated = fill_mask(imgs_masked, recreated_masks, pos, fill_size)
        display_imgs(
            torch.cat((imgs, imgs_masked, recreated)),
            n_columns=min(len(imgs), n_columns), ipy_display=ipy_display, delete_file=delete_file,
        )

    return hist


if __name__ == '__main__':
    main()

import datetime
import pickle
import os
from typing import Optional, Any
import numpy as np
from tqdm.auto import tqdm, trange  # type: ignore
from copy import deepcopy
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchinfo import summary

from torch_numbers.utils import count_params, display_examples
from torch_numbers.data import get_dataset, DATASET_OPTIONS
from torch_numbers.gan import Discriminator, Generator


def main(
        n_epochs: int = 50,
        pretrained_encoder: Optional[nn.Module] = None,
        pretrained_decoder: Optional[nn.Module] = None,
        batch_size: int = 64,
        lr_d: float = 0.001, betas_d: tuple[float, float] = (0.5, 0.99),
        lr_g: float = 0.01, betas_g: tuple[float, float] = (0.8, 0.99),
        interims_dim: int = 128,
        print_model_summary: bool = False,
        pbar_smoothing: float = 0.03,
        save_discr: Optional[str] = 'GAN_discr.pt',
        save_gen: Optional[str] = 'GAN_gen.pt',
        save_hist: Optional[str] = 'hist.pkl',
):
    train_loader, discr, gen, opt_d, opt_g = prepare(
        pretrained_encoder=pretrained_encoder,
        pretrained_decoder=pretrained_decoder,
        batch_size=batch_size,
        lr_d=lr_d, betas_d=betas_d,
        lr_g=lr_g, betas_g=betas_g,
        interims_dim=interims_dim,
        print_model_summary=print_model_summary,
    )

    crit_digit = nn.CrossEntropyLoss(label_smoothing=0.03)
    crit_val = nn.CrossEntropyLoss(label_smoothing=0.2)

    hist = train(
        train_loader, discr, gen, crit_digit, crit_val, opt_d, opt_g,
        n_epochs=n_epochs, pbar_smoothing=pbar_smoothing,
        ipy_display=False, delete_file=False,
    )
    if save_hist is not None:
        with open(save_hist, 'wb') as f:
            pickle.dump(hist, f)

    if save_discr is not None:
        torch.save(discr, save_discr)
    if save_gen is not None:
        torch.save(gen, save_gen)
    return discr, gen


def train_batch(discr: nn.Module, gen: nn.Module,
                crit_digit: nn.Module, crit_val: nn.Module,
                opt_d: Optional[torch.optim.Optimizer], opt_g: Optional[torch.optim.Optimizer],
                imgs: Tensor, lbls: Tensor,
                other_gen: Optional[nn.Module] = None,
                log: Optional[dict[str, Any]] = None):
    zeros = torch.zeros_like(lbls, dtype=torch.long)
    ones = torch.ones_like(lbls, dtype=torch.long)
    if other_gen is None:
        other_gen = gen

    # train generator
    if opt_g is None:
        fake_imgs = gen(lbls)
        loss_gen = torch.tensor(torch.nan)
    else:
        opt_g.zero_grad()
        fake_imgs = gen(lbls)
        validity, logits = discr(fake_imgs)

        loss_lbl = crit_digit(logits, lbls)
        loss_val = crit_val(validity, ones)
        loss_gen = loss_lbl + loss_val
        loss_gen.backward()
        opt_g.step()

    # train discriminator
    extra_fake = other_gen(lbls)
    all_imgs = torch.cat((imgs, fake_imgs.detach(), extra_fake.detach()))
    val_lbls = torch.cat((ones, zeros, zeros))
    if opt_d is None:
        validity, logits = discr(all_imgs)
        loss_discr = torch.tensor(torch.nan)
        loss_d_lbl = torch.tensor(torch.nan)
        loss_d_val = torch.tensor(torch.nan)
    else:
        opt_d.zero_grad()
        validity, logits = discr(all_imgs)

        loss_d_lbl = crit_digit(logits[:len(lbls)], lbls)
        loss_d_val = crit_val(validity, val_lbls)
        loss_discr = loss_d_lbl + loss_d_val
        loss_discr.backward()
        opt_d.step()

    # calculate various metrics
    pred_discr = torch.argmax(logits[:len(lbls)], dim=-1)
    pred_gen = torch.argmax(logits[len(lbls):2*len(lbls)], dim=-1)
    acc_discr = (pred_discr == lbls).sum().item() / len(lbls)
    acc_gen = (pred_gen == lbls).sum().item() / len(lbls)
    pred_valid_fake = torch.argmax(validity[len(lbls):2*len(lbls)], dim=-1)
    gen_fooled = (pred_valid_fake == ones).sum().item() / len(ones)
    pred_valid = torch.argmax(validity[:2*len(lbls)], dim=-1)
    acc_valid = (pred_valid == val_lbls[:2*len(lbls)]).sum().item() / len(pred_valid)
    if log is None:
        log = dict()
    log.update({
        'time': datetime.datetime.now(),
        'batch_size': len(imgs),
        'loss_gen': loss_gen.item(),
        'loss_discr': loss_discr.item(),
        'loss_d_lbl': loss_d_lbl.item(),
        'loss_d_val': loss_d_val.item(),
        'acc_discr': acc_discr,
        'err_discr': 1 - acc_discr,
        'acc_gen': acc_gen,
        'err_gen': 1 - acc_gen,
        'gen_fooled': gen_fooled,
        'acc_valid': acc_valid,
        'err_valid': 1 - acc_valid,
    })

    return log


def prepare(
        pretrained_encoder: Optional[nn.Module] = None,
        pretrained_decoder: Optional[nn.Module] = None,
        dataset: DATASET_OPTIONS = 'MNIST',
        interims_dim: Optional[int] = None,
        batch_size: int = 64,
        lr_d=0.001, betas_d=(0.5, 0.99),
        lr_g=0.01, betas_g=(0.8, 0.99),
        print_model_summary: bool = False,
):
    # data
    train_ds = get_dataset(dataset, normalize='sigmoid')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    if interims_dim is None:
        if pretrained_encoder is not None:
            interims_dim = pretrained_encoder(pretrained_encoder.example_input_array).shape[-1]
        else:
            interims_dim = 256

    # models
    discr = Discriminator(encoder=pretrained_encoder, encoded_dim=interims_dim, norm='instance')
    if print_model_summary:
        print(summary(discr, input_data=discr.example_input_array,
                      col_names=["input_size", "output_size", "num_params"]))
    else:
        print(f"Discriminator has {count_params(discr, trainable=True):,d} trainable parameters")

    gen = Generator(decoder=pretrained_decoder, num_classes=10, inter_dim=interims_dim, norm='instance')
    if print_model_summary:
        print(summary(gen, input_data=gen.example_input_array,
                      col_names=["input_size", "output_size", "num_params"]))
    else:
        print(f"Generator has {count_params(gen, trainable=True):,d} trainable parameters")

    # optimizer
    opt_d = torch.optim.Adam(discr.parameters(), lr=lr_d, betas=betas_d)
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr_g, betas=betas_g)

    return train_loader, discr, gen, opt_d, opt_g


def train(
        train_loader: DataLoader,
        discr: nn.Module,
        gen: nn.Module,
        crit_digit: nn.Module,
        crit_val: nn.Module,
        opt_d: torch.optim.Optimizer,
        opt_g: torch.optim.Optimizer,
        n_epochs: int = 5,
        load_gen_every: int = 1_000,
        store_gen_every: int = 10_000,
        store_gens_dir: str = './generators',
        pbar_smoothing: float = 0.03,
        ipy_display: bool = True,
        delete_file: bool = True,
) -> list[dict[str, Any]]:
    if not os.path.exists(store_gens_dir):
        os.mkdir(store_gens_dir)
    hist = []
    expw: dict[str, Any] = dict()
    n_examples: int = len(train_loader.sampler)  # type: ignore
    other_gen = deepcopy(gen)  # fix this generator - do not train
    last_gen_load = 0
    last_gen_store = 0
    for epoch in trange(n_epochs):
        with tqdm(total=n_examples, smoothing=pbar_smoothing) as pbar:
            for batch_idx, (imgs, lbls) in enumerate(train_loader):
                if pbar.n - last_gen_store >= store_gen_every:
                    last_gen_store = pbar.n
                    path = os.path.join(store_gens_dir, f"{pbar.n:06d}.pt")
                    torch.save(gen, path)
                if pbar.n - last_gen_load >= load_gen_every:
                    last_gen_load = pbar.n
                    choices = [m for m in os.listdir(store_gens_dir) if m.endswith('.pt')]
                    if len(choices):
                        path = os.path.join(store_gens_dir, np.random.choice(choices))
                        other_gen = torch.load(path)

                log: dict[str, Any] = {
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                }
                train_batch(discr, gen, crit_digit, crit_val, opt_d, opt_g, imgs, lbls, other_gen=other_gen, log=log)

                hist.append(log)

                for k, v in log.items():
                    if isinstance(v, (int, float)):
                        expw[k] = pbar_smoothing * v + (1 - pbar_smoothing) * expw.get(k, v)
                pbar.set_postfix_str(
                    "err_valid={err_valid:.2%}, err_digit={err_discr:.2%}, ".format(**expw)
                    + "err_digit_gen={err_gen:.2%}".format(**expw)  # noqa: W503
                )
                pbar.update(len(imgs))

        display_examples(
            gen, n_lines=3, n_columns=10, ipy_display=ipy_display, delete_file=delete_file,
            path=f"fake_images_{epoch:03d}.png",
        )

    return hist


if __name__ == '__main__':
    main()

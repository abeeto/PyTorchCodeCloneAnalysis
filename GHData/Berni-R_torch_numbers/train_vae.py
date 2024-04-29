import datetime
import pickle
from typing import Optional, Any
from tqdm.auto import tqdm, trange  # type: ignore
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchinfo import summary

from torch_numbers.utils import count_params, display_imgs
from torch_numbers.data import get_dataset, DATASET_OPTIONS
from torch_numbers.vae import VAEEncoder, VAEDecoder, Sampling, vae_losses


def main(
        n_epochs: int = 50,
        batch_size: int = 64,
        latent_dim: int = 30,
        interims_dim: int = 128,
        beta: Optional[float] = None,
        alpha: float = 1.0,
        print_model_summary: bool = False,
        pbar_smoothing: float = 0.03,
        save_encoder: Optional[str] = 'VAE_Encoder.pt',
        save_decoder: Optional[str] = 'VAE_Decoder.pt',
        save_hist: Optional[str] = 'hist.pkl',
):
    train_loader, encoder, decoder, sampler, z_to_digit, opt = prepare(
        batch_size=batch_size,
        latent_dim=latent_dim,
        interims_dim=interims_dim,
        print_model_summary=print_model_summary,
    )

    if beta is None:
        beta = latent_dim / 28**2
    hist = train(
        train_loader, encoder, decoder, sampler, z_to_digit, opt,
        n_epochs=n_epochs, beta=beta, alpha=alpha, pbar_smoothing=pbar_smoothing,
        ipy_display=False, delete_file=False,
    )
    if save_hist is not None:
        with open(save_hist, 'wb') as f:
            pickle.dump(hist, f)

    if save_encoder is not None:
        torch.save(encoder, save_encoder)
    if save_decoder is not None:
        torch.save(decoder, save_decoder)
    return encoder, decoder


def train_batch(encoder: nn.Module, decoder: nn.Module, sampler: nn.Module, z_to_digit: nn.Module,
                opt: torch.optim.Optimizer, imgs: Tensor, lbls: Tensor, log: Optional[dict[str, Any]] = None,
                beta: float = 0.01, alpha: float = 0.1):
    opt.zero_grad()

    mu, log_var = encoder(imgs)
    z = sampler(mu, log_var)
    reconstruction = decoder(z)

    digits = z_to_digit(z)

    reconstr_loss, kl_loss = vae_losses(reconstruction, imgs, mu, log_var)
    digit_loss = F.cross_entropy(digits, lbls, label_smoothing=0.05)
    loss = reconstr_loss + beta * kl_loss + alpha * digit_loss

    loss.backward()
    opt.step()

    digits = torch.argmax(digits, dim=1)
    digit_acc = (digits == lbls).sum() / len(lbls)

    if log is None:
        log = dict()
    log['time'] = datetime.datetime.now()
    log['batch_size'] = len(imgs)
    log['reconstr_loss'] = reconstr_loss.item()
    log['kl_loss'] = kl_loss.item()
    log['digit_loss'] = digit_loss.item()
    log['digit_acc'] = digit_acc.item()
    log['digit_err'] = 1 - digit_acc.item()
    log['loss'] = loss.item()

    return log


def prepare(
        dataset: DATASET_OPTIONS = 'MNIST',
        batch_size: int = 64,
        latent_dim: int = 10,
        interims_dim: int = 128,
        print_model_summary: bool = False,
):
    # data
    train_ds = get_dataset(dataset, normalize='sigmoid')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # models
    encoder = VAEEncoder(None, interims_dim=interims_dim, encoded_dim=latent_dim)
    if print_model_summary:
        print(summary(encoder, input_data=encoder.example_input_array,
                      col_names=["input_size", "output_size", "num_params"]))
    else:
        print(f"Encoder has {count_params(encoder, trainable=True):,d} trainable parameters")

    decoder = VAEDecoder(None, interims_dim=interims_dim, encoded_dim=latent_dim)
    if print_model_summary:
        print(summary(decoder, input_data=decoder.example_input_array,
                      col_names=["input_size", "output_size", "num_params"]))
    else:
        print(f"Decoder has {count_params(decoder, trainable=True):,d} trainable parameters")

    inter_dim = latent_dim // 2 + 1
    z_to_digit = nn.Sequential(
        nn.Linear(latent_dim, inter_dim),
        nn.LeakyReLU(),
        nn.Linear(inter_dim, inter_dim),
        nn.BatchNorm1d(inter_dim),
        nn.LeakyReLU(),
        nn.Linear(inter_dim, inter_dim),
        nn.LeakyReLU(),
        nn.Linear(inter_dim, 10),
        nn.Softmax(dim=1),
    )
    if print_model_summary:
        print(summary(z_to_digit, input_data=torch.randn(2, latent_dim),
                      col_names=["input_size", "output_size", "num_params"]))
    else:
        print(f"Classifier has {count_params(z_to_digit, trainable=True):,d} trainable parameters")

    # optimizer
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + list(z_to_digit.parameters()),
        lr=0.001, betas=(0.9, 0.999),
    )

    return train_loader, encoder, decoder, Sampling(), z_to_digit, opt


def train(
        train_loader: DataLoader,
        encoder: nn.Module,
        decoder: nn.Module,
        sampler: nn.Module,
        z_to_digit: nn.Module,
        opt: torch.optim.Optimizer,
        n_epochs: int = 5,
        beta: float = 0.01,
        alpha: float = 0.1,
        pbar_smoothing: float = 0.03,
        ipy_display: bool = True,
        delete_file: bool = True,
) -> list[dict[str, Any]]:
    hist = []
    expw: dict[str, Any] = dict()
    n_examples: int = len(train_loader.sampler)  # type: ignore
    for epoch in trange(n_epochs):
        encoder.train()
        decoder.train()
        sampler.train()
        with tqdm(total=n_examples, smoothing=pbar_smoothing) as pbar:
            for batch_idx, (imgs, lbls) in enumerate(train_loader):
                log: dict[str, Any] = {
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                }
                train_batch(encoder, decoder, sampler, z_to_digit, opt, imgs, lbls, log=log, beta=beta, alpha=alpha)

                hist.append(log)

                for k, v in log.items():
                    if isinstance(v, (int, float)):
                        expw[k] = pbar_smoothing * v + (1 - pbar_smoothing) * expw.get(k, v)
                pbar.set_postfix_str(
                    "loss={loss:.4}, reconstr_loss={reconstr_loss:.4}, kl_loss={kl_loss:.4}, ".format(**expw)
                    + "digit_loss={digit_loss:.4}, err={digit_err:.2%}, ".format(**expw)  # noqa: W503
                )
                pbar.update(len(imgs))

        encoder.eval()
        decoder.eval()
        sampler.eval()
        n_columns = 30
        for imgs, lbls in train_loader:
            imgs = imgs[:n_columns]
            break
        mu, log_var = encoder(imgs)
        z = sampler(mu, log_var)
        reconstr = decoder(z)
        display_imgs(
            torch.cat((imgs, reconstr)),
            n_columns=min(len(imgs), n_columns), ipy_display=ipy_display, delete_file=delete_file,
        )

    return hist


if __name__ == '__main__':
    main()

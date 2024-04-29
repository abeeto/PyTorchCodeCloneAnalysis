from argparse import ArgumentParser

import pytorch_lightning as pl

from data import TorchTextDataModule
from model import LitSentiment


def parse_args(args=None):
    parser = ArgumentParser()
    parser = TorchTextDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args(args)


def main(args=None):

    # Tell me what to do!
    args = parse_args(args)

    # I'm, like...SO RANDOM!
    pl.seed_everything(42)

    # Data!
    dm = TorchTextDataModule()
    dm.setup()

    # Model!
    model = LitSentiment(
        vocab_size=len(dm.vocab), embed_dim=32, num_class=len(dm.labels),
    )

    # Train!
    pl.Trainer.from_argparse_args(args).fit(model, dm)

if __name__ == '__main__':
    main()

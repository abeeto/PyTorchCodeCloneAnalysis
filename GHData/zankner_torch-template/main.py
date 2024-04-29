import os
import argparse
import pytorch_lightning as pl

from model import Model
from loggers import BlankLogger
import data

SAVE_DIR = "ADD SAVE DIR!!!"
PROJECT_NAME = "mnist"
PROJECT_ENTITY = "zackankner"


def train(args):
    pl.seed_everything(args.seed)

    logger = pl.loggers.WandbLogger(project=PROJECT_NAME,
                                    entity=PROJECT_ENTITY,
                                    name=args.model_name)

    callbacks = [
        pl.callbacks.ModelCheckpoint(dirpath=os.path.join(
            SAVE_DIR, args.model_name),
                                     every_n_epochs=args.save_freq,
                                     save_top_k=-1),
        pl.callbacks.TQDMProgressBar(refresh_rate=20)
    ]

    model = Model(args)

    train_loader = data.get_train_loader()
    val_loader = data.get_val_loader()

    trainer = pl.Trainer(gpus=args.gpus,
                         check_val_every_n_epoch=args.eval_freq,
                         max_epochs=args.epochs,
                         logger=logger,
                         callbacks=callbacks)
    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)


def test(args):
    logger = BlankLogger()

    checkpoint = os.path.join(SAVE_DIR, args.checkpoint)

    model = Model.load_from_checkpoint(checkpoint)

    test_loader = data.get_test_loader()

    trainer = pl.Trainer(gpus=args.gpus, logger=logger)
    trainer.test(model, dataloaders=test_loader)


def main(args):
    if args.action == "train":
        train(args)
    elif args.action == "test":
        test(args)
    else:
        raise ValueError("Must either be in train or test mode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General args
    parser.add_argument("--gpus", nargs="+", default=None)

    sp = parser.add_subparsers(dest="action")

    # Training hparams
    train_parser = sp.add_parser("train")
    train_parser.add_argument("--model-name", type=str, required=True)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--save-freq", type=int, default=1)
    train_parser.add_argument("--eval-freq", type=int, default=1)
    train_parser.add_argument("--resume-path", type=str, default=None)
    train_parser.add_argument("--epochs", type=int, required=True)
    train_parser.add_argument("--batch-size", type=int, required=True)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.0)
    train_parser.add_argument("--step-size", type=float, default=20)
    train_parser.add_argument("--lr-decay", type=float, default=0.1)

    # Testing hparams
    test_parser = sp.add_parser("test")
    test_parser.add_argument("--checkpoint", type=str, required=True)
    test_parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()

    main(args)
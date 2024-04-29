import random

import numpy as np
import pytorch_lightning as pl
import torch
from absl import app, flags

from data_module import MNISTDataModule
from models import GAN

FLAGS = flags.FLAGS
flags.DEFINE_integer("n_batch", 256, "batch size")
flags.DEFINE_integer("seed", 7, "random seed")
flags.DEFINE_float("lr", 0.001, "learning rate")

DEVICE = torch.device(("cuda" if torch.cuda.is_available() else "cpu"))
if DEVICE == "cuda":
    accelerator = "gpu"
    GPUS = min(1, torch.cuda.device_count())

else:
    accelerator = "cpu"
    GPUS = None


def main(argv):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    data = MNISTDataModule(batch_size=FLAGS.n_batch)
    model = GAN(lr=FLAGS.lr)
    model.plot_imgs()
    trainer = pl.Trainer(max_epochs=100, gpus=GPUS, accelerator=accelerator)
    trainer.fit(model, data)
    model.plot_imgs()


if __name__ == "__main__":
    app.run(main)

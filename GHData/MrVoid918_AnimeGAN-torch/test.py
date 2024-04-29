# import sys
# sys.path.append('./model')      #Quick way to bypass
'''
from model.discriminator import Discriminator
from model.generator import Generator
import init_train
'''

from typing import List, Optional
from typer import Argument, Option
import typer
import numpy as np
import torch
import torch.nn as nn
from trial import Trial
from pathlib import Path


def main(batch_size: int = Option(32, "-b"),
         G_lr: float = Option(0.01, help="Learning rate for generator training"),
         D_lr: float = Option(0.01, help="Learning rate for discriminator training"),
         GAN_G_lr: float = Option(0.00008, help="Learning rate for GAN generator training"),
         GAN_D_lr: float = Option(0.00016, help="Learning rate for GAN discriminator training"),
         G_epoch: int = Option(10, help="Iteration of generator training"),
         D_epoch: int = Option(3, help="Iteration of discriminator training"),
         GAN_epoch: int = Option(1, help="Iteration of GAN training"),
         itr: int = Option(1, help="Iteration of whole NOGAN training"),
         optim_type: str = Option("ADAB", help="Options of Optimizers for Generator"),
         level: str = Option("O1", help="FP16 Training Flags, Refer to Apex Docs"),
         adv_weight: float = Option(1.0, help="Weight for discriminator loss"),
         load_dir: Optional[Path] = Option(None)):
    """
    NOGAN training Trial.
    """
    torch.backends.cudnn.benchmark = True
    assert(itr > 0), "Number must be bigger than 0"
    trial = Trial(batch_size=batch_size, G_lr=G_lr, D_lr=D_lr, optim_type=optim_type, level=level)
    decreased_lr = False
    if load_dir is not None:
        trial.load_trial(load_dir)
    for _ in range(itr):
        trial.Generator_NOGAN(epochs=G_epoch, content_weight=3.0, recon_weight=10.,
                              loss=['content_loss', 'recon_loss'],)
        trial.Discriminator_NOGAN(epochs=D_epoch, adv_weight=adv_weight)
        trial.GAN_NOGAN(GAN_epoch, GAN_G_lr=GAN_G_lr, GAN_D_lr=GAN_D_lr, adv_weight=adv_weight)

import argparse
import json
import random
import tempfile
from datetime import datetime
from pathlib import Path

import torch
import wandb

from config import RANDOM_STATE
from utils import __WANDB_CLOSE__, __WANDB_OFFLINE__, __WANDB_ONLINE__

PROJECT_NAME = "..."
ONLINE = True
RANDOM_STATE = 42


__WANDB_ONLINE__ = "online"
__WANDB_OFFLINE__ = "offline"
__WANDB_CLOSE__ = "close"


class Config:
    def __init__(self) -> None:
        self.conf = {
            "~lr": 0.001,
            "~batch_size": 128,
            "~epochs": 200,
            "~early_stopping_patience": 3,
            "~optimizer": "adam",
            "~loss": "mse",
        }
        self.wandb_conf = {
            "project": PROJECT_NAME,
            "entity": "hzzz",
            # mkdir /wandb/PROJECT_NAME
            "dir": f"/wandb/{PROJECT_NAME}",
            "mode": "online" if ONLINE else "offline",
        }

    def init(self, args):
        if args.exp_file:
            self.conf = json.load(open(args.exp_file))
        self.wandb_conf["mode"] = args.wandb
        self.checkpoints = args.checkpoints

    @property
    def wandb_enable(self):
        return self.wandb_conf["mode"] != __WANDB_CLOSE__

    @property
    def cuda_rank(self):
        if self.distributed:
            return torch.distributed.get_rank()
        return 0

    def init_wandb(self):
        if self.wandb_enable != __WANDB_CLOSE__:
            wandb.init(config=self.conf, **self.wandb_conf)

    def log(self, json):
        if self.wandb_enable != __WANDB_CLOSE__:
            wandb.log(json)

    def wandb_finish(self):
        if self.wandb_enable != __WANDB_CLOSE__:
            wandb.finish()

    def __getattr__(self, name: str):
        if self.wandb_enable != __WANDB_CLOSE__:
            return config[name]
        else:
            return self.conf[name]

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __str__(self) -> str:
        return str(self.conf)


def prep_env():
    timestamp = int(datetime.timestamp(datetime.now()))

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_file", type=str)
    parser.add_argument(
        "--checkpoints", type=str, default=f"checkpoints/checkpoints_{timestamp}"
    )

    parser.add_argument(
        "--wandb",
        type=str,
        default="offline",
        help=f"{__WANDB_ONLINE__}, {__WANDB_OFFLINE__}, {__WANDB_CLOSE__}",
    )
    namespace, extra = parser.parse_known_args()

    Path(namespace.checkpoints).mkdir(exist_ok=True)
    print("Checkpoints:", namespace.checkpoints)
    config.init(namespace)
    try:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(torch.distributed.get_rank())
        config.distributed = True
    except:
        config.distributed = False

    return namespace


config = Config()

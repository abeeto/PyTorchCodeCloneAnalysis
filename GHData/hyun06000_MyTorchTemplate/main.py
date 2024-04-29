import argparse
from datetime import datetime


import wandb


from sweep_config.config_generator import get_sweep_config
from trainer.trainer import train


def get_parser():
    parser = argparse.ArgumentParser(description='Mode selection for training')
    parser.add_argument("--sweep_config", type=str, default="bayes",
        help='sweep config. one of random, grid or bayes.(default[str] : bayes) '
    )

    return parser

def main(parser):

    args = parser.parse_args()
    sweep_config = get_sweep_config(args.sweep_config)
    now = datetime.now().isoformat(timespec='seconds').replace(":","-")

    wandb.login()
    sweep_id = wandb.sweep(
    project=f"simple-classification-sweep-{args.sweep_config}-[{now}]",
    sweep = sweep_config
    )

    count = 10 # number of runs to execute
    wandb.agent(sweep_id, function=train, count=count)

if __name__ == "__main__":
    parser = get_parser()
    main(parser)
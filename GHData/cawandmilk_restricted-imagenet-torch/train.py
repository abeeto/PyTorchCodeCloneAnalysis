import torch
import transformers

import argparse
import pprint

from src.dataset import RestrictedImagenetDataset
from src.model import get_model
from src.trainer import RestrictedImageNetTrainer
from src.utils import get_transforms


def define_argparser(is_continue: bool = False):
    p = argparse.ArgumentParser()

    p.add_argument(
        "--train_path",
        type=str,
        default="data/train",
    )
    p.add_argument(
        "--valid_path",
        type=str,
        default="data/val",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )
    p.add_argument(
        "--warmup_ratio", 
        type=float, 
        default=.2,
    )
    p.add_argument(
        "--n_epochs",
        type=int,
        default=10,
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default="ckpt",
    )
    p.add_argument(
        "--gpu_id", 
        type=int, 
        default=-1,
    )
    p.add_argument(
        "--verbose", 
        type=int, 
        default=2,
    )
    p.add_argument(
        "--max_grad_norm",
        type=float,
        default=5.,
        help="Threshold for gradient clipping. Default=%(default)s",
    )
    p.add_argument(
        "--use_radam", 
        action="store_true",
    )

    config = p.parse_args()
    return config


def get_datasets(config):
    ## Get datasets.
    tr_ds = RestrictedImagenetDataset(config.train_path, transforms=get_transforms(is_train=True))
    vl_ds = RestrictedImagenetDataset(config.valid_path, transforms=get_transforms(is_train=False))

    ## Get dataloaders.
    tr_loader = torch.utils.data.DataLoader(
        tr_ds, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
    )
    vl_loader = torch.utils.data.DataLoader(
        vl_ds, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
    )

    return tr_loader, vl_loader


def get_optimizer(config, model):
    if config.use_radam:
        optimizer = torch.optim.RAdam(model.parameters(), lr=config.lr)
    else:
        raise AssertionError()

    return optimizer


def main(config: argparse.Namespace):
    def print_config(config):
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(vars(config))
    print_config(config)

    ## Get loader.
    tr_loader, vl_loader = get_datasets(config)

    print(
        f"|train| = {len(tr_loader)}",
        f"|valid| = {len(vl_loader)}",
    )

    n_total_iterations = len(tr_loader) * config.n_epochs
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print(
        f"# total iters = {n_total_iterations}",
        f"# warmup iters = {n_warmup_steps}",
    )

    ## Get pretrained model with specified softmax layer.
    model = get_model(pretrained=False)
    optimizer = get_optimizer(config, model)

    ## We will not use our own loss.
    crit = None
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        n_warmup_steps,
        n_total_iterations,
    )

    if config.gpu_id >= 0:
        # model.device("cuda")
        model.cuda(config.gpu_id)

    ## Start train.
    trainer = RestrictedImageNetTrainer(config)
    model = trainer.train(
        model,
        crit,
        optimizer,
        scheduler,
        tr_loader,
        vl_loader,
    )
    

if __name__ == "__main__":
    config = define_argparser()
    main(config)

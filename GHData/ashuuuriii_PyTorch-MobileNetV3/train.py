import argparse
import yaml
from shutil import copyfile
import time
from datetime import datetime
import os
import sys
import logging
import argparse

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from ptflops import get_model_complexity_info

from model import MobileNetV3
from utils import EMA


def train(
    epochs,
    train_loader,
    val_loader,
    model,
    loss_fn,
    optimizer,
    output_dir,
    device,
    ema_decay=None,
    tf_logger=None,
    log_interval=10,
):
    ema = EMA(model, ema_decay) if ema_decay else None
    top_val_acc = 0.0

    for i in range(epochs):
        logging.info(f"Epoch {i}\n-----------------------------")
        train_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            device,
            i,
            ema,
            tf_logger,
            log_interval,
        )
        val_acc = validate(val_loader, model, loss_fn, device, i, ema, tf_logger)

        if val_acc > top_val_acc:
            model_pth = os.path.join(output_dir, "best.pth")
            torch.save(model.state_dict(), model_pth)
            top_val_acc = val_acc

        model_pth = os.path.join(output_dir, f"epoch_{i}.pth")
        torch.save(model.state_dict(), model_pth)

    endtime = datetime.fromtimestamp(time.time())
    logging.info(f"Training finished at {endtime.strftime('%d/%m/%y-%H:%M:%S')}")

    if tf_logger:
        tf_logger.close()


def train_epoch(
    dataloader,
    model,
    loss_fn,
    optimizer,
    device,
    epoch,
    ema=None,
    tf_logger=None,
    log_interval=10,
):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, gt) in enumerate(dataloader):
        X, gt = X.to(device), gt.to(device)

        pred = model(X)  # forward
        loss = loss_fn(pred, gt)  # compute loss

        # backpropagate loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema:
            ema.update()

        # log loss and weights
        loss, current = loss.item(), batch * len(X)
        accuracy = (pred.argmax(1) == gt).type(torch.float).sum().item()
        accuracy = accuracy / pred.shape[0]
        logging.info(
            f"train accuracy(top-1): {accuracy:>7f}, train loss: {loss:>7f} [{current:>5d}/{size:>5d}]"
        )

        if tf_logger and batch % log_interval == 0:
            tf_logger.add_scalar(f"train/{epoch}/loss", loss, batch)
            tf_logger.add_scalar(f"train/{epoch}/accuracy", accuracy, batch)
            for name, module in model.named_modules():
                if hasattr(module, "weight"):
                    tf_logger.add_histogram(f"weight/{name}", module.weight, batch)
                elif hasattr(module, "bias"):
                    tf_logger.add_histogram(f"bias/{name}", module.bias, batch)


def validate(dataloader, model, loss_fn, device, epoch, ema=None, tf_logger=None):
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    val_loss, correct = 0, 0

    with torch.no_grad():
        model.eval()
        if ema:
            ema.apply()

        for X, gt in dataloader:
            X, gt = X.to(device), gt.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, gt).item()
            correct += (pred.argmax(1) == gt).type(torch.float).sum().item()

    # log loss and accuracy
    val_loss /= n_batches
    accuracy = correct / size
    logging.info(
        f"val accuracy(top-1): {(100*accuracy):>0.7f}%, avg loss: {val_loss:>8f}\n"
    )

    if ema:
        ema.restore()

    if tf_logger:
        tf_logger.add_scalar("val/loss", val_loss, epoch)
        tf_logger.add_scalar("val/accuracy", accuracy, epoch)

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, required=True, help="Path to yaml config file"
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force PyTorch to use cpu when CUDA is available.",
    )
    args = parser.parse_args()

    # Load config file
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
        settings = cfg["settings"]
        train_cfg = cfg["train_params"]
        opt_cfg = cfg["optimizer_params"]
        data_cfg = cfg["dataset_params"]
        model_cfg = cfg["model_params"]

    # Set up text logger, TensorBoard logging and logging directory.
    ts = time.time()
    ts = datetime.fromtimestamp(ts)
    dir_ts = ts.strftime("%d%m%H%M%S")

    output_dir = os.path.join(settings["output_dir"], dir_ts)
    cfg_filename = args.cfg.split("/")[-1]
    os.makedirs(output_dir, exist_ok=True)
    copyfile(args.cfg, os.path.join(output_dir, cfg_filename))

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "train.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )

    tf_logger = SummaryWriter(log_dir=output_dir)

    # Set device and random seed
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed(settings["seed"])

    # Create the dataloader
    augs = {
        "train": transforms.Compose(
            [
                transforms.RandomCrop(data_cfg["img_size"]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(data_cfg["mean"], data_cfg["sd"]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(data_cfg["mean"], data_cfg["sd"]),
            ]
        ),
    }

    train_set = datasets.CIFAR100(
        root=data_cfg["path"], train=True, download=True, transform=augs["train"]
    )
    val_set = datasets.CIFAR100(
        root=data_cfg["path"], train=False, download=True, transform=augs["val"]
    )

    train_loader = DataLoader(
        train_set, batch_size=data_cfg["batch_size"], shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_set, batch_size=data_cfg["batch_size"], shuffle=False)

    # Initialise the model
    model = MobileNetV3(
        model_size=model_cfg["model_size"],
        n_classes=data_cfg["classes"],
        head_type=model_cfg["classification_head"],
        initialisation=model_cfg["initialisation_type"],
        drop_rate=model_cfg["drop_out_probability"],
        alpha=model_cfg["width_multiplier"],
    )
    model = model.to(device)

    img_size = data_cfg["img_size"]
    macs, params = get_model_complexity_info(
        model, (3, img_size, img_size), print_per_layer_stat=False
    )

    logging.info(
        f"MobileNetV3-{model_cfg['model_size']}, at {model_cfg['width_multiplier']}x"
    )
    logging.info(f"macs: {macs}, params: {params}")

    # Initialise the optimizer
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=opt_cfg["lr"],
        momentum=opt_cfg["momentum"],
        weight_decay=opt_cfg["weight_decay"],
    )

    schdeduler = optim.lr_scheduler.StepLR(
        optimizer, opt_cfg["lr_decay_step"], gamma=opt_cfg["lr_decay"]
    )

    loss_fn = nn.CrossEntropyLoss()  # standard loss function for classification

    # Start training
    logging.info(
        f"Starting training at {ts.strftime('%d/%m/%y-%H:%M:%S')} on {device}."
    )
    train(
        train_cfg["epochs"],
        train_loader,
        val_loader,
        model,
        loss_fn,
        optimizer,
        output_dir,
        device,
        ema_decay=train_cfg["ema"],
        tf_logger=tf_logger,
        log_interval=settings["log_interval"],
    )

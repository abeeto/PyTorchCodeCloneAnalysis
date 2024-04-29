import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.datasets as datasets
import numpy as np

from loguru import logger
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler


def get_device(no_cuda=False):
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info("Using {}\n", device)
    return device


def get_idx_to_class(class_to_idx:dict):
    idx_to_class = {}
    for key, val in class_to_idx.items():
        idx_to_class[val] = key
    return idx_to_class


def make_dir(args):
    if not os.path.exists(args.ckpts_dir):
        os.mkdir(args.ckpts_dir)
    if not os.path.exists(args.imgs_dir):
        os.mkdir(args.imgs_dir)
    if not os.path.exists(args.logs_dir):
        os.mkdir(args.logs_dir)


def train(
    model,
    device,
    criterion,
    optimizer,
    train_loader,
):
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)

        acc = pred.eq(target.view_as(pred)).sum().item() / len(data)
        train_acc += acc

        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    return train_loss, train_acc


def validate(
    model,
    device,
    criterion,
    val_loader
):
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            val_acc += pred.eq(target.view_as(pred)).sum().item() / len(data)
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    return val_loss, val_acc


def get_dataloader(
    batch_size,
    test_batch_size,
    train_dataset,
    valid_dataset,
    test_dataset,
    num_workers,
    valid_split,
):
    num_train = len(train_dataset)
    idx = list(range(num_train))
    np.random.shuffle(idx)
    split = int(valid_split * num_train)
    train_idx, valid_idx = idx[split:], idx[:split]
    train_dataset = Subset(train_dataset, train_idx)
    valid_dataset = Subset(valid_dataset, valid_idx)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    return train_loader, valid_loader, test_loader


def get_CIFAR10loader(
    batch_size,
    test_batch_size,
    download,
    root,
    valid_split,
    augmenter=None,
):
    list_transforms = [
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    if augmenter:
        transform=T.Compose(augmenter + list_transforms)
    else:
        transform=T.Compose(list_transforms)
    test_transform = T.Compose(list_transforms)

    train_dataset = datasets.CIFAR10(
        root=root,
        train=True,
        transform=transform,
        download=download,
    )
    valid_dataset = datasets.CIFAR10(
        root=root,
        train=True,
        transform=test_transform,
        download=download,
    )
    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        transform=test_transform,
        download=download,
    )
    train_loader, val_loader, test_loader = get_dataloader(
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        num_workers=4,
        valid_split=valid_split,
    )
    return train_loader, val_loader, test_loader


def fit(
    model: nn.Module,
    crit,
    epochs,
    init_lr,
    ckpt_file,
    train_loader,
    val_loader,
):
    device = get_device()
    model = model.to(device)
    train_loss, train_acc, val_loss, val_acc = [[] for _ in range(4)]
    optimizer = optim.SGD(model.parameters(), lr=init_lr)
    for epoch in range(1, epochs+1):
        # training
        train_loss_, train_acc_ = train(
            model=model,
            device=device,
            criterion=crit,
            optimizer=optimizer,
            train_loader=train_loader,
        )
        train_loss.append(train_loss_)
        train_acc.append(train_acc_)

        # validation
        val_loss_, val_acc_ = validate(
            model=model,
            device=device,
            criterion=crit,
            val_loader=val_loader,
        )
        val_loss.append(val_loss_)
        val_acc.append(val_acc_)

        logger.info(f"Train Epoch: {epoch} / {epochs} LR: {optimizer.param_groups[0]['lr']:.6f}")
        logger.info(f"Train Loss: {train_loss_:.5f}\tTrain Accuracy: {train_acc_:.5f}")
        logger.info(f"Valid Loss: {val_loss_:.5f}\tValid Accuracy: {val_acc_:.5f}\n")

    logger.info(f"Saving model to {ckpt_file}\n")
    torch.save(model.state_dict(), ckpt_file)
    history = {
        "train_history": {"train_accuracy": train_acc, "train_loss": train_loss},
        "val_history": {"val_accuracy": val_acc, "val_loss": val_loss},
    }
    return history


def getMNISTloader(
    root,
    download,
    batch_size,
    test_batch_size,
    valid_split,
):
    transform=T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        transform=transform,
        download=download,
    )
    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        transform=transform,
        download=download,
    )
    train_loader, val_loader, test_loader = get_dataloader(
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_workers=4,
        valid_split=valid_split,
    )
    return train_loader, val_loader, test_loader


def getFashionMNISTloader(
    root,
    download,
    batch_size,
    test_batch_size,
    valid_split,
):
    transform=T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.FashionMNIST(
        root=root,
        train=True,
        transform=transform,
        download=download,
    )
    test_dataset = datasets.FashionMNIST(
        root=root,
        train=False,
        transform=transform,
        download=download,
    )
    train_loader, val_loader, test_loader = get_dataloader(
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_workers=4,
        valid_split=valid_split,
    )
    return train_loader, val_loader, test_loader



def adjust_lr_exp(
    optimizer: optim.Optimizer, 
    epoch: int,
    init_lr: float,
    gamma=0.8,
    milestone=5,
):
    if epoch < milestone:
        new_lr = init_lr
    else:
        new_lr = init_lr * math.exp(-gamma*(epoch-milestone))
    if optimizer.param_groups[0]["lr"] != new_lr:
        print(f"[INFO]Train epoch {epoch}: Adjusting learning rate to {new_lr:.7f}")
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr


def adjust_lr(
    optimizer: optim.Optimizer, 
    epoch: int,
    milestone1: int=15,
    milestone2: int=25,
):
    if epoch == milestone1 or epoch == milestone2:
        new_lr = optimizer.param_groups[0]["lr"] / 10
        logger.info(f"Train epoch {epoch}: Adjusting learning rate to {new_lr:.5f}\n")
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

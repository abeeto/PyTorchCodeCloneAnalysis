#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-01-21 14:09
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import torch
from torchvision import datasets, transforms


def get_dataloader(args):

    if args.dataset.lower() == 'mnist':

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_root, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Resize((32, 32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_root, train=False, download=True,
                           transform=transforms.Compose([
                               transforms.Resize((32, 32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)

    elif args.dataset.lower() == 'svhn':

        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(args.data_root, split='train', download=True,
                          transform=transforms.Compose([
                              transforms.Resize((32, 32)),
                              transforms.ToTensor(),
                              transforms.Normalize(
                                  (0.43768206, 0.44376972, 0.47280434), (0.19803014, 0.20101564, 0.19703615)),
                              # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                          ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(args.data_root, split='test', download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(
                                  (0.43768206, 0.44376972, 0.47280434), (0.19803014, 0.20101564, 0.19703615)),
                              # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                          ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)

    elif args.dataset.lower() == 'cifar10':

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_root, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_root, train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)

    else:
        raise ValueError(f"Unknown dataset name: {args.dataset}.")

    return train_loader, test_loader


class Namespace:
    def __init__(self, dataset="svhn", batch_size=256):
        dataset = dataset.lower()
        self.dataset = dataset
        self.data_root = None

        if dataset == "svhn":
            self.data_root = "~/.datasets/SVHN"
        elif dataset == "cifar10":
            self.data_root = "~/.datasets"

        self.batch_size = batch_size


def main():
    args = Namespace()
    train_loader, test_loader = get_dataloader(args)


if __name__ == "__main__":
    main()

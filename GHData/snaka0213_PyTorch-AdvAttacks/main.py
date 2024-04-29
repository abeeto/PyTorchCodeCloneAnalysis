#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse

from src.attack import attack
from src.mnist import mnist_loaders, mnist_model
from src.fgsm import fgsm
from src.pgd import pgd

def argparser(
    batch_size=50, path=None,
    attack=None, epsilon=0.1, n_iters=40):

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--path", default=path)

    # adversarial attack
    parser.add_argument("--attack", default=attack)

    # fgsm, pgd
    parser.add_argument("--epsilon", type=float, default=epsilon)

    # pgd
    parser.add_argument("--n_iters", type=int, default=n_iters)

    args = parser.parse_args()
    return args

def mean(l):
    return sum(l)/len(l)

if __name__ == "__main__":
    args = argparser()
    if args.path is None:
        raise ValueError("NotFound Path")

    model = mnist_model()
    checkpoint = torch.load(args.path)
    model.load_state_dict(checkpoint["state_dict"])
    _, test_loader = mnist_loaders(args.batch_size)
    opt = optim.Adam(model.parameters())

    # attack
    if args.attack == "fgsm":
        attack = fgsm
        kwargs = {
            "epsilon": args.epsilon
        }
    elif args.attack == "pgd":
        attack = pgd
        kwargs = {
            "epsilon": args.epsilon,
            "n_iters": args.n_iters
        }
    else:
        raise ValueError("Unknown attack")

    total_acc, total_acc_atk = attack(test_loader, model, **kwargs)
    print("Before Accuracy: {acc:.4f}, After Accuracy: {acc_atk:.4f}"\
        .format(acc=mean(total_acc), acc_atk=mean(total_acc_atk)))

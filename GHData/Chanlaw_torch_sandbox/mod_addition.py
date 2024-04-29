# %%%

import random
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from grokking_utils import full_loss, Transformer

alpha = 0.3
cuda = "cuda"


def gen_train_test(frac_train, num, seed=0):
    # Generate train and test split
    pairs = [(i, j, num) for i in range(num) for j in range(num)]
    random.seed(seed)
    random.shuffle(pairs)
    div = int(frac_train * len(pairs))
    return torch.tensor(pairs[:div]), torch.tensor(pairs[div:])


def train_model(
    p: int,
    frac_train: float,
    n_epochs: int,
    alpha: float = 1.0,
    record_every: int = 100,
    seed: int = 0,
    weight_decay: float = 1.0,
    cuda=None,
) -> Tuple[Transformer, List[float], List[float]]:
    model = Transformer(
        num_layers=1,
        d_vocab=p + 1,
        d_model=128,
        d_mlp=512,
        d_head=32,
        num_heads=4,
        n_ctx=3,
        act_type="ReLU",
    )
    train_losses: List[float] = []
    test_losses: List[float] = []

    for param in model.parameters():
        param.data = param.data * alpha

    train_data, test_data = gen_train_test(frac_train, p)
    train_labels = torch.tensor([(i + j) % p for i, j, p in train_data])
    test_labels = torch.tensor([(i + j) % p for i, j, p in test_data])

    if cuda is not None:
        cuda = torch.device(cuda)
        model.to(cuda)
        train_data = train_data.to(cuda)
        test_data = test_data.to(cuda)
        train_labels = train_labels.to(cuda)
        test_labels = test_labels.to(cuda)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=weight_decay, betas=(0.9, 0.98)
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(step / 10, 1)
    )

    for epoch in tqdm(range(n_epochs)):
        model(train_data)
        train_loss = full_loss(model, train_data, train_labels)
        test_loss = full_loss(model, test_data, test_labels)
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

        if epoch % record_every == 0:
            print(
                f"{epoch}_{np.log(train_loss.item()):.4f}_{np.log(test_loss.item()):.4f}"
            )

        train_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return model, train_losses, test_losses


# %%
if __name__ == "__main__":
    model, train_losses, test_losses = train_model(
        p=113,
        frac_train=0.3,
        n_epochs=40000,
        alpha=alpha,
        record_every=100,
        seed=0,
        cuda=cuda,
        weight_decay=0.0,
    )

    pd.dataframe(train_loss=train_losses, test_loss=test_losses).to_csv(
        f"train_{alpha}.csv"
    )

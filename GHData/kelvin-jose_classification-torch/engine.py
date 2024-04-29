import torch
import numpy as np
import torch.nn as nn


def loss_fn(logits, y):
    loss = nn.CrossEntropyLoss()
    output = loss(logits, y)
    return output


def accuracy(preds, target):
    return torch.tensor(torch.sum(preds == target).item() / len(preds))


def train_fn(data_loader, model, batch_size, optimizer, device):
    for ix, batch in enumerate(data_loader):
        X = torch.reshape(batch[0], (batch_size, -1))
        y = batch[1]
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ix % 1000 == 0:
            print(f'batch: {ix} loss: {loss}')


def valid_fn(data_loader, model, batch_size, device):
    epoch_acc = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for ix, batch in enumerate(data_loader):
            X = torch.reshape(batch[0], (batch_size, -1))
            y = batch[1]
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            softmax_score = softmax(logits)
            max_probs, preds = torch.max(softmax_score, dim=1)
            epoch_acc.append(accuracy(preds, y))
    print(f'validation accuracy: {np.mean(epoch_acc)}')


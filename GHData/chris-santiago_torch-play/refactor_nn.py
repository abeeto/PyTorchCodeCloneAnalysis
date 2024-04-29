import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from get_mnist import load_data

MNIST = load_data().clean(binarize=False)


def make_datasets(data):
    x_train, y_train, x_valid, y_valid = map(torch.tensor, tuple(data))
    return TensorDataset(x_train, y_train.long()), TensorDataset(x_valid, y_valid.long())


def make_dataloaders(data, batch_size):
    train, valid = make_datasets(data)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(valid, batch_size=batch_size)
    )


def accuracy(preds, actual):
    return (preds.argmax(1) == actual).float().mean()


class MnistLogistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)
        self.loss_func = nn.CrossEntropyLoss()  # includes log softmax activation

    def forward(self, xb):
        return self.lin(xb.float())


class MnistTrainer:
    def __init__(self, model, epochs, lr):
        self.model = model
        self.epochs = epochs
        self.optim = torch.optim.SGD(self.model.parameters(), lr=lr)

    def loss_batch(self, xb, yb, optim=None):
        loss = self.model.loss_func(self.model.forward(xb), yb)
        if optim:
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
        return loss.item(), len(xb)

    @staticmethod
    def get_total_loss(losses, nums):
        return np.sum(np.multiply(losses, nums)) / np.sum(nums)

    def fit(self, data, bs):
        train_dl, valid_dl = make_dataloaders(data, bs)
        for epoch in range(self.epochs):
            self.model.train()
            losses, nums = zip(*[self.loss_batch(xb, yb, optim=self.optim) for xb, yb in train_dl])
            train_loss = self.get_total_loss(losses, nums)
            self.model.eval()
            with torch.no_grad():
                losses, nums = zip(*[self.loss_batch(xb, yb) for xb, yb in valid_dl])
                val_loss = self.get_total_loss(losses, nums)
            print(f'Epoch: {epoch}. Training Loss: {train_loss}. Validation Loss: {val_loss}')

    def predict(self, x_valid):
        self.model.eval()
        with torch.no_grad():
            return self.model.forward(x_valid)


def main():
    model = MnistLogistic()
    learner = MnistTrainer(model, epochs=20, lr=0.1)
    learner.fit(MNIST, 256)


if __name__ == '__main__':
    main()

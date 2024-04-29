import contextlib
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from model import NeuralNetwork

"""
https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""


def train(device, dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"\rloss: {loss:>7f}  [{current:>5d}/{size:>5d}]", end='')
    print()


def test(device, dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")


@contextlib.contextmanager
def timeit(desc):
    _ts = time.time()
    yield
    _te = time.time()
    print(f"{desc}: {_te-_ts:.2f} sec")


def _main():
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 32

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    if device == "cpu":
        torch.set_num_threads(12)

    model = NeuralNetwork().to(device)
    print(model)

    with timeit("training"):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

        epochs = 80
        for t in range(epochs):
            with timeit(f"Epoch {t + 1}"):
                print(f"Epoch {t + 1}\n-------------------------------")
                train(device, train_dataloader, model, loss_fn, optimizer)
                test(device, test_dataloader, model, loss_fn)
            print("\n")
        print("Done!")

        torch.save(model.state_dict(), f"model-{batch_size}-{epochs}-{device}.pth")
        print("Saved PyTorch Model State to model.pth")


if __name__ == '__main__':
    _main()

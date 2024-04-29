# -*- coding:utf-8 -*-

import math
import numpy as np
import torch
from torch import nn
import torchvision.datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

ComputeDevice = "cuda" if torch.cuda.is_available() else "cpu"

training_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=torchvision.transforms.ToTensor(),
)

test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=torchvision.transforms.ToTensor(),
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

i_historyList = []
J_historyList = []

labelType = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

class NeuralNetworkModel(torch.nn.Module):
    def __init__(self):
        super(NeuralNetworkModel, self).__init__()

        self.LayerFunc = torch.nn.Linear(1, 1, device=ComputeDevice)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, lossfunc, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(ComputeDevice), y.to(ComputeDevice)
        pred = model(X)
        loss = lossfunc(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        J_historyList.append(loss.detach().item())
        i_historyList.append(epoch)


def test(dataloader, model, lossfunc):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(ComputeDevice), y.to(ComputeDevice)
            pred = model(X)
            test_loss += lossfunc(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    Model = NeuralNetworkModel()
    Model.to(device=ComputeDevice)

    Model.load_state_dict(torch.load(r".\FashionMNISTModel.pth"))

    LossFunc = torch.nn.CrossEntropyLoss()
    Optimizer = torch.optim.SGD(Model.parameters(), lr=1e-3)

    '''
    iternum = 5

    for epoch in range(iternum):
        train(train_dataloader, Model, LossFunc, Optimizer)
        test(test_dataloader, Model, LossFunc)

    torch.save(Model.state_dict(), r".\FashionMNISTModel.pth")


    plt.figure(1)
    plt.plot(i_historyList, J_historyList)
    plt.show()
    '''

    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(ComputeDevice), y.to(ComputeDevice)

            pred = Model(X)
            for i in range(4):
                print(f"-------------------{i}----------------------")
                print("It might be: ", labelType[pred.argmax(1)[i]])
                print("It actually is: ", labelType[y[i]])
                print("------------------------------------------")

                plt.clf()
                plt.imshow(np.reshape(X[i][0].cpu().numpy(), (28, 28)))
                plt.draw()
                plt.pause(1.5)

            break

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import model

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
max_epoch = 150

train_data = datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)
test_data = datasets.CIFAR100(
    root="./data", train=False, download=True, transform=transforms.ToTensor()
)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# initiate
network_PE = model.basic_MLP_with_PE([500, 500, 300, 300])
network_woPE = model.basic_MLP_without_PE([500, 500, 300, 300])
network_PE.to(device)
network_woPE.to(device)

optimizer_PE = torch.optim.SGD(network_PE.parameters(), lr=1e-4, momentum=0.999)
optimizer_woPE = torch.optim.SGD(network_woPE.parameters(), lr=1e-4, momentum=0.999)
loss_fn = nn.CrossEntropyLoss()

# test
def accuracy(network, test_loader):
    n_problem = 0
    n_correct = 0
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y_tensor = F.one_hot(y, num_classes=100)
        y_tensor = y_tensor.type(torch.float32)
        y = y_tensor.to(device)

        prediction = network(x)

        n_problem += y.shape[0]
        n_correct += torch.count_nonzero(
            torch.argmax(prediction, dim=1) == torch.argmax(y, dim=1)
        ).item()

    accuracy = n_correct / n_problem
    return accuracy


# train function
def train_network(network, train_loader, optimizer, loss_fn, max_epoch, is_pe):
    accuracys = []
    if is_pe:
        pe_norms = []
    for epoch in range(max_epoch):
        for x, y in train_loader:
            x = x.to(device)
            y_tensor = F.one_hot(y, num_classes=100)
            y_tensor = y_tensor.type(torch.float32)
            y = y_tensor.to(device)

            prediction = network(x)
            loss = loss_fn(prediction, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # statistics
        acc = accuracy(network, test_loader) * 100

        accuracys.append(acc)
        if is_pe:
            pe_norms.append(torch.norm(network.pe, "fro").item())
        
        print(f"epoch {epoch} accuracy {acc}%")

    if is_pe:
        return (accuracys, pe_norms)
    else:
        return accuracys


if __name__ == "__main__":
    PE_accuracys, pe_norms = train_network(
        network_PE, train_loader, optimizer_PE, loss_fn, max_epoch, True
    )
    woPE_accuracys = train_network(
        network_woPE, train_loader, optimizer_woPE, loss_fn, max_epoch, False
    )

    x = range(max_epoch)
    plt.figure(1)
    plt.plot(x, PE_accuracys, label="with PE")
    plt.plot(x, woPE_accuracys, label="without PE")
    plt.legend()
    plt.savefig("accuracy.png")

    plt.figure(2)
    plt.plot(x, PE_accuracys, label="accuracy with PE")
    plt.plot(x, pe_norms, label="norm of PE")
    plt.legend()
    plt.savefig("PE_informations.png")

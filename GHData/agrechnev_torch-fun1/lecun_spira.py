# By Oleksiy Grechnyev, IT-JIM, 29-Dec-2020

import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn

import lecun_plot


########################################################################################################################
def main():
    lecun_plot.set_default()

    device = torch.device('cuda:0')
    # device = torch.device('cpu')
    nn, nd, nc, nh = 1000, 2, 3, 100

    # Create data (spirals)
    x = torch.zeros(nn * nc, nd).to(device)
    y = torch.zeros(nn * nc, dtype=torch.int64).to(device)
    tt = np.linspace(0, 1, nn)
    for c in range(nc):
        ivar = np.linspace(2 * np.pi * c / nc, 2 * np.pi * (c + 2) / nc, nn) + np.random.randn(nn) * 0.2

        y[nn * c: nn * (c + 1)] = c
        tmp = np.vstack([tt * np.sin(ivar), tt * np.cos(ivar)]).T.astype('float32')
        x[nn * c: nn * (c + 1), :] = torch.from_numpy(tmp)

    # Plot data
    if False:
        lecun_plot.plot_data(x, y)
        plt.show()

    # Model
    model = torch.nn.Sequential(
        torch.nn.Linear(nd, nh),
        torch.nn.ReLU(),
        torch.nn.Linear(nh, nc),
    )
    model.to(device)
    print(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Train
    for t in range(10000):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        score, pred = torch.max(y_pred, 1)
        acc = (y == pred).sum().float() / len(y)
        print(f'Epoch : {t}, loss={loss.item()}, acc={acc}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Display
    lecun_plot.plot_model(x, y, model)
    plt.show()


########################################################################################################################
if __name__ == '__main__':
    main()

# By Oleksiy Grechnyev, 11-Jan-2021

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
    # Generate data
    x = torch.unsqueeze(torch.linspace(-1, 1, 101), dim=1).to(device)
    y = x**3 + 0.3*torch.rand(x.shape).to(device)
    if False:
        plt.scatter(x.cpu().numpy(), y.cpu().numpy())
        plt.axis('equal')
        plt.show()

    n_model = 5
    ns, nd, nc, nh = 1000, 1, 1, 100
    learning_rate = 1e-3
    lambda_l2 = 1e-5
    models = []
    for im in range(n_model):
        # MOdel
        model = torch.nn.Sequential(
            torch.nn.Linear(nd, nh),
            torch.nn.ReLU(),
            torch.nn.Linear(nh, nc),
        )
        model.to(device)
        print(model)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda_l2)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=lambda_l2)

        # Train
        for t in range(1000):
            y_pred = model(x)
            loss = criterion(y_pred, y)
            print(f'Epoch : {t}, loss={loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        models.append(model)

        if False:
            y_pred = model(x)
            plt.scatter(x.detach().cpu().numpy(), y.detach().cpu().numpy())
            plt.plot(x.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), 'r-', lw=5)
            plt.show()

    if True:
        plt.scatter(x.detach().cpu().numpy(), y.detach().cpu().numpy())
        for m in models:
            x_new = torch.unsqueeze(torch.linspace(-2, 2, 101), dim=1).to(device)
            y_pred_new = m(x_new)
            plt.plot(x_new.detach().cpu().numpy(), y_pred_new.detach().cpu().numpy(), 'r-', lw=2)
        plt.show()


########################################################################################################################
if __name__ == '__main__':
    main()
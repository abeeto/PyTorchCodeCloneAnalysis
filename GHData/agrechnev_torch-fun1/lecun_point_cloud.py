# By Oleksiy Grechnyev, IT-JIM, 26-Nov-2020

import sys

import numpy as np
import matplotlib.pyplot as plt

import torch


########################################################################################################################
def my_scatterplot(x, colors):
    x = x.cpu()
    x = x.numpy()
    colors = colors.cpu().numpy()
    plt.figure()
    plt.axis('equal')
    plt.scatter(x[:, 0], x[:, 1], c=colors, cmap='hsv', s=30)
    plt.axis('off')


########################################################################################################################
def main():
    # Set plt style
    plt.style.use(['dark_background', 'bmh'])
    plt.rc('figure', figsize=(10, 10))

    device = torch.device('cuda:0')
    n_points = 1000
    x = torch.randn(n_points, 2).to(device)
    colors = torch.atan2(x[:, 0], x[:, 1])

    while True:
        nh = 20
        model = torch.nn.Sequential(
            torch.nn.Linear(2, nh, bias=False),
            # torch.nn.ReLU(),
            torch.nn.Linear(nh, 2, bias=False),
        ).to(device=device)

        with torch.no_grad():
            y = model(x)
            my_scatterplot(y, colors)
        plt.show()


########################################################################################################################
if __name__ == '__main__':
    main()

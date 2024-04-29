import numpy as np
import torch

w = torch.tensor(torch.randn([3, 1]), requires_grad=True)

opt = torch.optim.Adam([w], lr=0.1)

def model(x):
    f = torch.stack([x * x, x, torch.ones_like(x)], 1)
    ## @ 矩阵相乘, * 矩阵点乘
    yhat = torch.squeeze(f @ w, 1)

    return yhat

def compute_loss(y, yhat):
    loss = torch.nn.functional.mse_loss(yhat, y)

    return loss

def generate_data():
    x = torch.randn(100) * 20 - 10
    y = 5 * x * x + 3

    return x, y

def train_step():
    x, y = generate_data()
    yhat = model(x)
    loss = compute_loss(y, yhat)

    opt.zero_grad()
    loss.backward()
    opt.step()

def main():
    for _ in range(4000):
        train_step()

    print(w.detach().numpy())

if __name__ == "__main__":
    main()


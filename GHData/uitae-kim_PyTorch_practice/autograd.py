import torch


if __name__ == "__main__":
    x = torch.tensor([1, 2, 3], dtype=torch.float, requires_grad=True)
    print(x)
    y = x ** 2
    y.retain_grad()
    print(y)
    v = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float)
    y.backward(v)
    print(y.grad)
    print(x.grad)

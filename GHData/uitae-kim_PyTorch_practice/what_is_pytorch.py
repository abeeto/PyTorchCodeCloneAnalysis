import torch


if __name__ == "__main__":
    r = torch.rand(5, 3)
    print(r)
    t2 = torch.empty(3, 5)
    t2 = r.reshape(torch.Size([3, 5]))
    print(t2)

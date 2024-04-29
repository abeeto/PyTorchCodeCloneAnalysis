import torch


def compute_z(a, b, c):
    r1 = torch.sub(a, b)
    r2 = torch.mul(r1, 2)
    z = torch.add(r2, c)
    return z


if __name__ == "__main__":
    print('Start!')
    print(torch.__version__)
    print('Scalar inputs:', compute_z(torch.tensor(1),
                                      torch.tensor(2),
                                      torch.tensor(3)))

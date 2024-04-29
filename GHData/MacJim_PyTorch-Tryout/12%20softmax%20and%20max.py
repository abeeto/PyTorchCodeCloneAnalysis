import torch
import torch.nn.functional as F


# MARK: torch.max
def test1():
    x = torch.Tensor(list(range(12)))
    x = x.view(3, 4)
    print("x:", x)
    print("x shape:", x.shape)

    # result1 = torch.max_pool1d(x, 1)
    # print(result1)

    result2 = torch.max(x, 1)
    print(result2[0])
    print(result2[1])


def test2():
    a = torch.Tensor([1, 2, 3, 4] * 2)
    a = a.view(2, 2, 2)
    print("a:", a)

    b1, b2 = torch.max(a, 0)
    print("b1:", b1)
    print("b2:", b2)

    c1, c2 = torch.max(a, 1)
    print("c1:", c1)
    print("c2:", c2)

    d1, d2 = torch.max(a, 2)
    print("d1:", d1)
    print("d2:", d2)

    e = F.softmax(a, 0)
    print("e:", e)    # 0.5 * 8

    f1, f2 = torch.max(e, 0)
    print("f1", f1)
    print("f2", f2)


# MARK: F.softmax
def test3():
    a = torch.Tensor([1] * 4 + [2] * 4 + [3] * 4)
    a = a.view(3, 4)
    print("a:", a)

    # Elements in `dim` add up to 1.
    b = F.softmax(a, 0)
    print("b:", b)    # 0.09, 0.2447, 0.6652

    c = F.softmax(a, 1)
    print("c:", c)    # 0.25 * 12

    d = F.softmax(a, 2)
    print("d:", d)    # IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)


def test4():
    a = torch.Tensor([1, 2, 3, 4] * 2)
    a = a.view(2, 2, 2)
    print("a:", a)

    b = F.softmax(a, 0)
    print("b:", b)    # 0.5 * 8

    c = F.softmax(a, 1)
    print("c:", c)

    d = F.softmax(a, 2)
    print("d:", d)


test2()
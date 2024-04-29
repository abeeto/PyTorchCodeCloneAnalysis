# first Tutorial - What is PyTorch: https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py
from __future__ import print_function
import torch
print(torch.cuda.is_available())
# tensor 2x4 bedeutet: 2 Reihe, 4 Spalten (2 row 4 columns)
# tensor 2x4 beispiel: [[x, x, x, x],
#                       [x, x, x, x]]


def partOne():
    x = torch.empty(1, 4)
    print(x)

    y = torch.rand(5, 3)
    print(y)

    z = torch.zeros(5, 3, dtype=torch.long)
    print(z)

    a = torch.tensor([5.5, 3])
    print(a)

    # or create tensor based on existing tensor
    a = a.new_ones(5, 3, dtype=torch.double)  # new_* take in sizes
    print(a)

    # override a again
    a = torch.rand_like(a, dtype=torch.float)
    print(a)

    print(a.size())

    b = torch.tensor([[2, 2, 2],
                      [2, 2, 2],
                      [2, 2, 2],
                      [2, 2, 2],
                      [2, 2, 2]], dtype=torch.float)
    print(b)
    print(b.dtype)

    # addition in 2 ways
    # 1 syntax
    # print(a + b)

    # 2 syntax
    # print(torch.add(a, b))

    # providing output tensor as argument
    result = torch.empty(b.size())  # size will be expanded if needed?
    print("result while empty: " + "\n" + str(result))
    torch.add(a, b, out=result)  # out specifies where the result should be written to
    print("result after addition of a and b\n" + str(result))

    # in-place addition
    # note: every action with an _ changes the tensor!

    b.add_(a)
    print(b)  # b is no longer 2s!

    # Numpy-like indexing
    print(b[:, 1])  # print second column
    print(b[0, :])  # print first row


def partTwo():
    # Resizing/reshaping a tensor using torch.view
    c = torch.randn(4, 4)
    d = c.view(16)  # reshape to a 1 dimensional tensor with 16x1
    e = c.view(-1,
               8)  # reshape to 2 dimensions with 8 x 2; the size of -1 is inferred (gefolgert) from the other dimension
    f = c.view(1, 16)  # difference between [16] and [1, 16] ?

    print(c, c.size())
    print(d, d.size())
    print(e, e.size())
    print(f, f.size())

    # get python value
    x = torch.randn(1)
    print(x)
    print(x.item())


def partThree():
    # numpy bridge
    a = torch.ones(5)
    print("this is a: " + str(a))

    b = a.numpy()
    print("this is b: " + str(b))

    a.add_(1)
    print("this is a after the addition of 1 in each column: " + str(a))
    print("this is b after the addition of 1 in the a tensor, but b is tied to a by memory: " + str(b))

    # converting numpy array to Torch Tensor
    import numpy as np
    a = np.ones(5)
    print("this is a befor calc: " + str(a))
    b = torch.from_numpy(a)
    np.add(a, 1, out=a)
    print("this is a after calc: " + str(a))
    print("this is b: " + str(b))


def partFour():
    # CUDA Tensors
    # tensors can be moved onto any device unsing .to method

    # let us run this cell only if CUDA is available
    # use torch.device objects to move tensors in and out of GPU
    x = torch.randn(1)
    print("x in the beginning: " + str(x))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        y = torch.ones_like(x, device=device)
        x = x.to(device)
        z = x + y
        print("this is z (x + y): " + str(z))
        print("z to cpu and change dtype: " + str(z.to("cpu", torch.double)))


# partOne()
# partTwo()
# partThree()
# partFour()

a = torch.ones([2, 4])
print("a:", a)
print("a size:", a.size())
print("a view(1, -1):", a.view(1, -1))

print()     # blank line

t = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("t:", t)
print("t size:", t.size())
# print(t[0][0][0].item())

print()     # blank line

e = torch.flatten(t, start_dim=1)
print("e:", e)
print("e size:", e.size())

print()     # blank line

f = t.flatten(start_dim=0, end_dim=1)
print("f:", f)
print("f size:", f.size())

# Tutorial - What is PyTorch end


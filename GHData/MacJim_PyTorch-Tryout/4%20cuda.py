# Source: https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html

import torch


x = torch.ones(3, 3)

if (torch.cuda.is_available()):
    device = torch.device("cuda")  # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    # x = x.to(device)  # or just use strings ``.to("cuda")``
    x = x.cuda()    # This works as well.
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))  # ``.to`` can also change dtype together!

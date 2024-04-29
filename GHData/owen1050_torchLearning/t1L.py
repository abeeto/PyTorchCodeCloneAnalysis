from __future__ import print_function
import torch

x = torch.randn(5,5)
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device = device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))


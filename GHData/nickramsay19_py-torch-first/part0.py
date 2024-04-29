from __future__ import print_function
import torch
import numpy as np

# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!


x = torch.ones(5, 3, dtype=torch.long)
y = torch.ones(5, 3, dtype=torch.long)

# convert to numpy object
j = x.numpy()
print(j)


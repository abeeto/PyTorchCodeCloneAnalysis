import torch
import numpy as np

x = np.array([[1.,2.], [3.,4.]])
x = torch.from_numpy(x)
x.requires_grad = True
print(x)

y = x**2
print(y)

z = y.mean()
print(z)

z.backward()
print(x.grad)
print(x/2)
import torch
from torch_err import *

def f(x):
    return torch.exp(x[:,0]) + x[:,1]**2

test = torch.randn(10, 2, requires_grad=True)
test_err = torch.randn(10, 2, requires_grad=True)
test_y = f(test)
test_y_err = error(test, test_y, test_err)
print(test_y_err)
test_y_err = ferror(f, test, test_err)
print(test_y_err)

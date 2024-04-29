import numpy as np
import torch
x=torch.tensor(3.)
w=torch.tensor(4.,requires_grad=True)
b=torch.tensor(5.,requires_grad=True)

y=w*x+b
y.backward() #to take y derivative
print("dy/dw",w.grad)

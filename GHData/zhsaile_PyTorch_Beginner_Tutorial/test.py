import torch
import torch.nn as nn

x = torch.tensor(1.)
y = torch.tensor(2.)
w = torch.tensor(1., requires_grad=True)

# forward pass

y_hat = w*x

loss = (y_hat - y)**2

    print(loss)

# backward

loss.backward()
#print(w.grad)

# update weights

# Gradient calculate with autograd



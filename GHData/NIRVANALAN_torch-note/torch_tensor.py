import time

import torch
import torchvision
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda')

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

w1 = torch.randn(D_in, H, device=device)
w2 = torch.randn(H, D_out, device=device)

lr = 1e-6
T = 500
t1 = time.time()
for t in range(T):
    # forward passing
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    # compute
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())
    # backprop
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # update
    w1 += -lr * grad_w1
    w2 += -lr * grad_w2
print('time used:', time.time() - t1)

# create tensor
# V_data = [1, 2., 3.]
# V = torch.Tensor(V_data)

# x = autograd.Variable(torch.Tensor(V_data), requires_grad=True)
# print(x.data)
#
# y = autograd.Variable(torch.Tensor([4, 5, 6]), requires_grad=True)
# print((x + y).data)
#
# data = autograd.Variable(torch.randn(2, 2))
# print(data)
# print(F.relu(data))

# data = autograd.Variable(torch.randn(5))
# print(data)
# print(F.softmax(data, dim=0))
# print(F.softmax(data, dim=0).sum())
# print(F.log_softmax(data, dim=0))

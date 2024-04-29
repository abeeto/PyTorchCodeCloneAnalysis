import torch
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms


#L1S1
x = torch.tensor(1.,requires_grad=True)
w = torch.tensor(2.,requires_grad=True)
b = torch.tensor(3.,requires_grad=True)

y = w * x + b

y.backward()

print(x.grad)
print(w.grad)
print(b.grad)

# what if I change x.req_grad = false?

# if I just write:
#   x.requires_grad=True
#   y.backward()
# It will report error:
#   Trying to backward through the graph a second time,
#   but the buffers have already been freed. Specify retain_graph=True when calling backward the first time.
# Maybe I can only create a new x,w,b,y

x = torch.tensor(1.,requires_grad=False)
w = torch.tensor(2.,requires_grad=True)
b = torch.tensor(3.,requires_grad=True)

y = w * x + b

y.backward()

print(x.grad)
print(w.grad)
print(b.grad)

#L1S2

x = torch.randn(10,3)
y = torch.randn(10,2)

linear = nn.Linear(3,2)

print ('w: ', linear.weight)
print ('b: ', linear.bias)

ctn = nn.MSELoss()
opt = torch.optim.SGD(linear.parameters(),lr = 0.01)

pred = linear(x)
loss = ctn(pred,y)

print('loss: ', loss.item())

ll = loss.item()

loss.backward()

print('dL/dw: ', linear.weight.grad)
print('dL/db: ', linear.bias.grad)

opt.step()

pred = linear(x)
loss = ctn(pred, y)
print('loss after 1 step optimization: ', loss.item())

# now I want to train it until loss is less than 0.01, will I become successful?

# stp = 1
# while loss.item() > 0.01:
#     stp+=1
#     loss.backward()
#     opt.step()
#     pred = linear(x)
#     loss = ctn(pred, y)
#     if stp%10000 == 0:
#         print('loss after %d step optimization: '%stp, loss.item())
#
# print("I success!")

# you see that I comment out my code TAT, it doesn't converge at all!

# I immediately realize that it can't converge to less than 0.01 obviously, but why it can't converge?
# maybe I shouldn't use SGD for such small parameter, and I write GD with decrease learning rate by myself

stp = 1

while abs(loss.item()-ll) > 0.000001:
    stp+=1
    ll = loss.item()
    loss.backward()
    lrt = 1/stp
    if lrt < 0.001:
        lrt = 0.001
    linear.weight.data.sub_(lrt * linear.weight.grad.data)
    linear.bias.data.sub_(lrt * linear.bias.grad.data)
    pred = linear(x)
    loss = ctn(pred, y)
    if stp%100 == 0:
        print('loss after %d step optimization: '%stp, loss.item())

print("I success!")

#OK, now I success


#L1S3

x = np.array([[1, 2], [3, 4]])
y = torch.from_numpy(x)
z = y.numpy()

#L1S4


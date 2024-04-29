import torch
import numpy as np
#input array
x= torch.tensor([[73.0,67,43],[91,88,64],[87,134,58],[102,43,37],[69,96,70]])
x.shape
#output array
y=torch.tensor([[56.0,70],[81,101],[119,133],[22,37],[103,119]])
y.shape
w = torch.randn(2,3,requires_grad=True)

b = torch.randn(2,requires_grad=True)
print(w.shape, b.shape)

##Linear regression
def model(x):
    return torch.mm(x, torch.t(w))+b
    #return x@w.t()+b

##Loss funtion
def mse(preds, y):
    diff= preds*y
    return torch.sum(diff*diff)/diff.numel()
    

for i in range(100):
    pred = model(x)
    loss = mse(pred, y)
    loss.backward()
    if i==10:
        print(pred)
        print(loss)
        
   
    with torch.no_grad():
        w -= w.grad* 1e-5
        b -= b.grad* 1e-5
        if i==10:
            print(w)
            print(b)
        w.grad.zero_()
        b.grad.zero_()
    pred = model(x)
    loss = mse(pred, y)
    print(y)
    print(pred)
    
    
    
    

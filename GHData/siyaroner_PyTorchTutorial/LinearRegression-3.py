import torch
import numpy as np
#inputs: Temp(F),Rainfall(mm), Humidity(%)
inputs=np.array([[73,67,43],
                 [91,88,64],
                 [87,134,58],
                 [102,43,37],
                 [69,96,70]],dtype="float32")
#targets: apple(ton), Oranges(ton)
targets=np.array([[56,70],
                  [81,101],
                  [119,133],
                  [22,37],
                  [103,119]],dtype="float32")

inputs=torch.from_numpy(inputs)
targets=torch.from_numpy(targets)
# print(inputs,"\n",targets)
#apple_harvest=w11*x1+w12*x2+w13*x3+b1
#orenge_harvest=w21*x1+w22*x2+w23*x3+b2

w=torch.randn(2,3,requires_grad=True)
b=torch.randn(2,requires_grad=True)

# print(w,"\n",b)

def model (x):
    return x@w.t()+b

preds=model(inputs)
print(preds)

def mse(real,preds):
    diff=real-preds
    return torch.sum(diff*diff)/diff.numel()

loss=mse(targets[0],preds)
print("first: ",loss)

loss.backward() # allow us to get gradients for any feature we want 
# print(w)
# print(w.grad)
# print(b)
# print(b.grad)
# now we have to make to w and b gradient as zero to avoid miscalculation
# w.grad.zero_()
# b.grad.zero_()
# print(w.grad)
# print(b.grad)
# with torch.no_grad(): #allow us to take gradients continuously
#     w-=w.grad*1e-5
#     b-=b.grad*1e-5
#     w.grad.zero_()
#     b.grad.zero_()
# preds=model(inputs)
# loss=mse(targets,preds)
# print("second: ",loss) 

#let get this in a loop

for i in range (1000):
    preds2=model(inputs)
    loss2=mse(targets[0],preds2)
    loss2.backward()
    with torch.no_grad():
        w-=w.grad*1e-5
        b-=b.grad*0.0001
        w.grad.zero_()
        b.grad.zero_()

print(preds2)
print("last: ",loss2)











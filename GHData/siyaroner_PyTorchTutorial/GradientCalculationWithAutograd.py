import torch
import numpy as np
# x=torch.randn(4, requires_grad=True)
# y=x+2
# print(y) # grad_fn= addbackward
# z=y**2*2
# print(z) #grad_fn= mulbackward
# # z=z.mean() #without this z.backward() will give an error. Go to v vector
# print(z) # grad_fn=meanbackward
# v= torch.tensor([0.1,1.0,0.001,0.01], dtype=torch.float32) # to avoid an error we should create vector jacobion product.
#                                                       # and its size should be same with x
# z.backward(v) #dz/dy*dy/dx= dz/dx
# print(x.grad)

#to prevent python tracking gradients there are 3 options for this
# 1- x.requeres_grad_(False)
# 2-  x.detach
# 3-  with torch.no_grad()
# # 1:
# x.requires_grad_(False)
# print(x)
# #2:
# y=x.detach()
# print(y)
#3:
# with torch.no_grad():
#     y=x+2
#     print(y)

weights=torch.ones(4,requires_grad=True)
# way 1
# for epoch in range(1):
#     model_output=(weights*3).sum()
#     model_output.backward()
#     print(weights.grad) #every loop this will increase according the range. To prevent this we should empty grad
#     weights.grad.zero_() #so this will empty the grad
#way 2
optimizer=torch.optim.SGD(weights, lr=0.01) #SGD= Stokastik Gradient Decent, lr= learning rate
optimizer.step() 
optimizer.zero_grad() #it will empty grad
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

#tensors created by python lists
V_data=[1.,2.,3.]

V=torch.tensor(V_data)

print(V)

m_data=[[1.,2.,3.],[4.,5.,6.]]
m=torch.Tensor(m_data)

print(m)

t_data=[[[1,2],[3,4]],[[5,6],[7,8]]]

t=torch.tensor(t_data)

print(t)

print(V[0].item())

print(t[0])

a_data=[1,2,3,4,5,6]
a=torch.LongTensor(a_data)

print(a)
print(a.size())


#random data

x=torch.randn(3,4,5)
print(x)

#concatenating(rows)
x_1=torch.randn(2,5)
print(x_1)


y_1=torch.randn(3,5)
print(y_1)

z_1=torch.cat([x_1,y_1])
print(z_1)

#concatenating columns
x_2=torch.randn(2,3)
print(x_2)
print(">>>>>>>>")

y_2=torch.randn(2,5)
print(y_2)
print(">>>>>>>")
z_2=torch.cat([x_2,y_2],1)

print(z_2)



#requires grad
x=torch.tensor([1,2,3],requires_grad=True)

y=torch.tensor([4,5,6],requires_grad=True)

z=x+y

print(z)

print(z.grad_fn)

s=z.sum()
print(s)

print(s.grad_fn)

s.backward()
print(x.grad)

s.backward()
print(x.grad)

s.backward()
print(x.grad)







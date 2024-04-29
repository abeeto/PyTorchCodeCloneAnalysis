#using the autograd to calculate the gradient
import torch
from torch.autograd import Variable

dtype=torch.FloatTensor

N,D_in,H,D_out=64,1000,100,10

#create random Tensor to hold input and outputs, and warp tehm in Variables
x=Variable(torch.randn(N,D_in).type(dtype),requires_grad=False)
y=Variable(torch.randn(N,D_out).type(dtype),requires_grad=False)

#create random Tensors for weights,and wrap them in Variable
w1=Variable(torch.randn(D_in,H).type(dtype),requires_grad=True)
w2=Variable(torch.randn(H,D_out).type(dtype),requires_grad=True)

learning_rate=1e-6

for t in range(500):
	#forward pass:compute predicted y using operations on Variables
	y_pred=x.mm(w1).clamp(min=0).mm(w2)

	#compute and print loss using operations on Variables
	loss=(y_pred-y).pow(2).sum()
	print(t,loss)

	#use autograd to compute the backward pass.This call will compute the gradient
	#of loss with respect to all Variables with requires_grad=True
	loss.backward()
	#update the weights using gradient descent:w1.data and w2.data ate Tensors
	w1.data-=learning_rate*w1.grad.data
	w2.data-=learning_rate*w2.grad.data

	#manually zers the gradients after updating weights
	w1.grad.data.zero_()
	w2.grad.data.zero_()
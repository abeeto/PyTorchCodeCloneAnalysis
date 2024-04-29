"""
AUTO GRAD PACKAGE
Gradient calculation package used for optimization of a model
"""

import torch
## To find the gradient of some function with respect to x
# we must set requires_grad=True while creating our tensor
x = torch.rand(3, requires_grad=True)
print("x =", x)

## Whenever we do an operation on this tensor PyTorch creates a computational graph
# gradient is calulated using back propagation
# grad_fn is an argument that tells us the type of function done on the function y
y = x+2 
print("y =", y)
z = y*y*2 
print("z =", z)
w = z.mean()
print("w =", w)
"""
output:-
x = tensor([0.8086, 0.4025, 0.5344], requires_grad=True)
y = tensor([2.8086, 2.4025, 2.5344], grad_fn=<AddBackward0>)
z = tensor([15.7762, 11.5439, 12.8468], grad_fn=<MulBackward0>)
w = tensor(13.3890, grad_fn=<MeanBackward0>)
"""

## To calculate the gradient:
# saved intermediate values are freed as soon as backwards() function is called
# to retain the previous values retain_graph=True argument is passed
w.backward(retain_graph=True) #dw/dx 
print("grad dw/dz =", x.grad)

## Vector Jacobian matrix is created with the partial derivatives
# which is multiplied with the gradient vector to get the final result - Chain Rule
# we dont pass any argument to the backward() function which is implicitly 
# created for scalar values
# since z.mean() returns a scalar value we dont pass a gradient to be multiplied
# otherwise we'll have to give it the gradien argument - vector of same size.

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v, retain_graph=True)
print("grad dz/dx =", x.grad)

## To prevent PyTorch from maintaining history of gradients and calculating grad_fn
# When computing weights this calculations shouldnt be part of the computation
# Three Methods: 1). x.required_grad_(False)
#                2). x.detach()
#                3). with torch.no_grad()

## 1). x.required_grad_(False)
# note: notice that it is an inplace function
x.requires_grad_(False)
print("x =", x)

## 2). x.detach()
# k has same values but doesnt require gradient
k = x.detach()
print("k =", k)

## 3). with torch.no_grad()
with torch.no_grad():
    k = x+2
    print("k =", k)

## Whenever backward() is called the gradient for this tensor will be 
# accumulated into the .grad() attribute
# the values will be summed up 

weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print("weights.grad =", weights.grad)
    """
    output:-
    weights.grad = tensor([3., 3., 3., 3.])
    weights.grad = tensor([6., 6., 6., 6.])
    weights.grad = tensor([9., 9., 9., 9.])
    """
    ## To prevent the summing as shown in the above output
    # we use the weights.grad.zero_() function to replace the vleus with zero inplace
    weights.grad.zero_()
    """
    output:-
    weights.grad = tensor([3., 3., 3., 3.])
    weights.grad = tensor([3., 3., 3., 3.])
    weights.grad = tensor([3., 3., 3., 3.])
    """

## PyTorch built in optimizers
# E.g. stocastic gradient descent
# optimizer = torch.optim.SGD(weights, lr=0.01)
# optimizer.step()
# optimizer.zero_grad()
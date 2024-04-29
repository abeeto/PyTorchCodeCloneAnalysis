"""
UNDERSTAND PYTORCH CODE IN 10 MINUTES

PyTorch consists of 4 main packages:
1. torch: 			a general purpose array library similar to Numpy that can do computations on
                    GPU when the tensor type is cast to (torch.cuda.TensorFloat)
2. torch.autograd: 	a package for building a computational graph and automatically obtaining
                    gradients
3. torch.nn: 		a neural net library with common layers and cost functions
4. torch.optim: 	an optimization package with common optimization algorithms like SGD,Adam, etc

"""


##########################################################################################
# 0. IMPORT STUFF

import torch 							# arrays on GPU
import torch.autograd as autograd 		# build a computational graph
import torch.nn as nn 					# neural net library
import torch.nn.functional as F 		# most non-linearities are here
import torch.optim as optim 			# optimization package


##########################################################################################
# 1. TORCH

# Two matrices of size 2x3 into a 3d tensor 2x2x3
d = [[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [11., 12., 13.]]]
d = torch.Tensor(d)  # array from python list
print("shape of the tensor:", d.size())

# the first index is the depth
z = d[0] + d[1]
print("adding up the two matrices of the 3d tensor:", z)

# a heavily used operation is reshaping of tensors using .view()
print(d.view(2, -1))  # -1 makes torch infer the second dim


##########################################################################################
# 2. TORCH.AUTOGRAD

# d is a tensor not a node, to create a node based on it:
x = autograd.Variable(d, requires_grad=True)
print("the node's data is the tensor:", x.data.size())
print("the node's gradient is empty at creation:", x.grad)  # the grad is empty right now

# do operation on the node to make a computational graph
y = x + 1
z = x + y
s = z.sum()
print(s.creator)

# calculate gradients
s.backward()
print("the variable now has gradients:", x.grad)


##########################################################################################
# 3. TORCH.NN

# linear transformation of a 2x5 matrix into a 2x3 matrix
linear_map = nn.Linear(5, 3)
print("using randomly initialized params:", linear_map.parameters)


# data has 2 examples with 5 features and 3 target
data = torch.randn(2, 5)  # training
y = autograd.Variable(torch.randn(2, 3))  # target
# make a node
x = autograd.Variable(data, requires_grad=True)
# apply transformation to a node creates a computational graph
a = linear_map(x)
z = F.relu(a)
o = F.softmax(z)
print("output of softmax as a probability distribution:", o.data.view(1, -1))

# loss function
loss_func = nn.MSELoss()  # instantiate loss function
L = loss_func(z, y)  # calculateMSE loss between output and target
print("Loss:", L)


class Log_reg_classifier(nn.Module):
    def __init__(self, in_size, out_size):
        super(Log_reg_classifier, self).__init__()  # always call parent's init
        self.linear = nn.Linear(in_size, out_size)  # layer parameters

    def forward(self, vect):
        return F.log_softmax(self.linear(vect))


##########################################################################################
# 4. TORCH.OPTIM


# define optimizer
# instantiate optimizer with model params + learning rate
optimizer = optim.SGD(linear_map.parameters(), lr=1e-2)

# epoch loop: we run following until convergence
optimizer.zero_grad()  # make gradients zero
L.backward(retain_variables=True)
optimizer.step()
print(L)


# define model
model = Log_reg_classifier(10, 2)

# define loss function
loss_func = nn.MSELoss()

# define optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-1)

# send data through model in minibatches for 10 epochs
for epoch in range(10):
    for minibatch, target in data:
        model.zero_grad()  # pytorch accumulates gradients, making them zero for each minibatch

        # forward pass
        out = model(autograd.Variable(minibatch))

        # backward pass
        L = loss_func(out, target)  # calculate loss
        L.backward()  # calculate gradients
        optimizer.step()  # make an update step

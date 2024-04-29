
#---------Tensors

# Warm-up numpy

import numpy as np
N,D_in,H,D_out=64,1000,100,10
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
x=np.random.rand(N,D_in)
y=np.random.rand(N,D_out)
w1=np.random.randn(D_in,H)
w2=np.random.randn(H,D_out)
learning_rate=1e-6
for t in range(500):
    h=x.dot(w1)
    h_relu=np.maximum(h,0)
    y_pred=h_relu.dot(w2)
    loss=np.square(y_pred-y).sum()
    print(t,loss)
    
    grad_y_pred=2.0 * (y_pred-y)
    grad_w2=h_relu.T.dot(grad_y_pred)
    
    grad_h_relu=grad_y_pred.dot(w2.T)
    grad_h=grad_h_relu.copy()
    grad_h[h<0]=0
    grad_w1=x.T.dot(grad_h)
    
    w1-=learning_rate * grad_w1
    w2-=learning_rate * grad_w2
    
# pytorch:tensors
    
import torch
dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
N, D_in, H, D_out = 64, 1000, 100, 10
x=torch.randn(N,D_in).type(dtype)
x=torch.randn(D_in,D_out).type(dtype)
w1=torch.randn(D_in,H).type(dtype)
w2=torch.randn(H,D_out).type(dtype)
learning_rate=1e-6
for t in range(500):
    h=x.mm(w1)
    h_relu=h.clamp(min=0)
    y_pred=h_relu.mm(w2)
    loss=(y_pred-y).pow(2).sum()
    print(t,loss)
    
    grad_y_pred=2.0*(y_pred-y)
    grad_w2=h_relu.t().mm(grad_y_pred)
    
    grad_h_relu=grad_y_pred.mm(w2.t())
    grad_h=grad_h_relu.clone()
    grad_h[h<0]=0
    grad_w1=x.t().mm(grad_h)
    
    w1-=learning_rate * grad_w1
    w2-=learning_rate * grad_w2


#------------Autograd
    
# variables and autograd

import torch
from torch.autograd import Variable
dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
    
# Defining new autograd functions

import torch
from torch.autograd import Variable
class MyRelu(torch.autograd.Function):
    def forward(ctx,input):#ctx is a context object
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    @staticmethod
    def backward(ctx,grad_output):
        input, =ctx.save_tensors
        grad_input=grad_output.clone()
        grad_input[input<0]=0
        return grad_input
dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # To apply our Function, we use Function.apply method. We alias this as 'relu'.
    relu = MyReLU.apply

    # Forward pass: compute predicted y using operations; we compute
    # ReLU using our custom autograd operation.
    y_pred = relu(x.mm(w1)).mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # Use autograd to compute the backward pass.
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
    
# Static Graphs

import tensorflow as tf
import numpy as np

N,D_in,H,D_out=64,1000,100,10
x=tf.placeholder(tf.float32,shape=(None,D_in))
y=tf.placeHolder(tf.float32,shape=(None,D_out))
w1=tf.Variable(tf.random_normal((D_in,H)))
w2=tf.Variable(tf.random_normal((H,D_out)))
h=tf.matmul(x,w1)
h_relu=tf.maximum(h,tf.zeros(1))
y_pred=tf.matmul(h_relu,w2)
loss=tf.reduce_sum((y_pred-y)**2.0)
grad_w1,grad_w2=tf.gradients(loss,[w1,w2])
learning_rate=1e-6
new_w1=w1.assign(w1-learning_rate*grad_w1)
new_w2=w2.assign(w2-learning_rate*grad_w2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_value=np.random.randn(N,D_in)
    y_value=np.random.randn(N,D_out)
    for _ in range(500):
        loss_value, _, _=sess.run([loss,new_w1,new_w2],feed_dict={x:x_value,y:y_value})
        print(loss_value)
       
# nn module

import torch
from torch.autograd import Variable
N,D_in,H,D_out=64,1000,100,10
x=Variable(torch.randn(N,D_in))
y=Variable(torch.randn(N,D_out),requires_grad=False)
model=torch.nn.Sequential(
        torch.nn.Linear(D_in,H),
        torch.nn.ReLU(),
        torch.nn.linear(H,D_out),
    )
loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate=1e-4
for t in range(500):
    y_pred=model(x)
    loss=loss_fn(y_pred,y)
    print(t,loss.data[0])
    model.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.data-=learning_rate * param.grad.data

# optim
        
import torch
from torch.autograd import Variable
N,D_in,H,D_out=64,1000,100,10
x=Variable(torch.randn(N,D_in))
y=Variable(torch.randn(N,D_out),requires_grad=False)
model=torch.nn.Sequential(
        torch.nn.Linear(D_in,H),
        torch.nn.ReLU(),
        torch.nn.Linear(H,D_out),
    )
loss_fn=torch.nn.MSELoss(size_average=False)
learning_rate=1e-4
optimizer=torch.nn.optim.Adam(model.paramters(),lr=learning_rate)
for t in range(500):
    y_pred=model(x)
    loss=loss_fn(y_pred,y)
    print(t,loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
# Custom nn Modules

import torch
from torch.autograd import Variable

class TwoLayerNet(torch.nn.Module):
    def __init__(self,D_in,H,D_out):
        super(TwoLayerNet,self).__init__()
        self.linear1=torch.nn.Linear(D_in,H)
        self.linear2=torch.nn.Linear(H,D_out)
    def forward(self,x):
        h_relu=self.linear1(x).clamp(min=0)
        y_pred=self.linear2(h_relu)
        return y_pred
N,D_in,H,D_out = 64,1000,100,10
x=Variable(torch.randn(N,D_in))
y=Variable(torch.randn(N,D_out))
model=TwoLayerNet(D_in,H,D_out)
loss_fn=torch.nn.MSELoss(size_average=False)
optimizer=torch.optim.SGD(model.parameters(),lr=1e-4)
for t in range(500):
    y_pred=model(x)
    loss=loss_fn(y_pred,y)
    print(t,loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# control Flow + weight sharing

import random
import torch
from torch.autograd import Variable

class DynamicNet(torch.nn.Module):
    def __init__(self,D_in,H,D_out):
        super(DynamicNet,self).__init__()
        self.input_linear=torch.nn.Linear(D_in,H)
        self.middle_linear=torch.nn.Linear(H,H)
        self.output_linear=torch.nn.linear(H,D_out)
    def forward(self,x):
        h_relu=self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0,3)):
            h_relu=self.middle_linear(h_relu).clamp(min=0)
        y_pred=self.output_linear(h_relu)
        return y_pred
N, D_in, H, D_out = 64, 1000, 100, 10
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)
model = DynamicNet(D_in, H, D_out)
loss_fn=torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4,momentum=0.9)
for t in range(500):
     y_pred = model(x)
     loss = loss_fn(y_pred, y)
     print(t, loss.data[0])
     optimizer.zero_grad()
     loss.backward()
     optimizer.step()


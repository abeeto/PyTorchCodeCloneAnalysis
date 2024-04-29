# February 22 2019
import torch
import matplotlib.pyplot as plt
import numpy
import csv
import math
import time

start = time.time()

def getdata():
    rates = []
    filename = 'gbp_2001_2018.in'
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            q = []
            for item in row:
                q.append(float(item))
            rates.append(q)
            line_count += 1
        ntenors = int(rates[0][0])
        nlines = int(rates[1][0])
        rates.pop(0)
        rates.pop(0)
        rates.pop(0)
        rmax  =  numpy.amax(rates)
        rates = rates / rmax
    return ntenors, nlines, numpy.array(rates)

def make_xy(ntenors,  N, rates,sample_size,n_forcast):
    n_samples =   N - sample_size - n_forcast + 1
    x_in =  numpy.empty([n_samples,sample_size,ntenors])
    y_out = numpy.empty([n_samples,ntenors])
    for n in range(n_samples):
        for i in range(sample_size):
            x_in[n,i,:] = rates[n+i,:]
        y_out[n,:] = rates[n+sample_size+n_forcast-1, :]
    return n_samples, x_in, y_out

def ex1():
    ntenors, nlines, rates = getdata()
# PARAMETERS
# N is batch size;
# D_in is input dimension;
# H is hidden dimension;
# D_out is output dimension.
    N, sample_size, n_forcast = 501, 24, 1       #  21, 2, 1
    D_in, H, D_out = 7, 20, 7                   #  7, 20, 7
    learning_rate = 1e-4                        #   5e-5
    M =  100    # number of iterations          #  1000
    chart_type = "loglog"                       # "logy"   "logx"  "loglog" ""
#
    n_samples, x_in, y_out =  make_xy(ntenors, N, rates,sample_size,n_forcast)
    # x_in  is numpy array: [n_samples,sample_size,ntenors];
    # y_out is numpy array: [n_samples,ntenors];
    xn = torch.from_numpy(x_in )        # convertion from numpy to pytorch tensor
    yn = torch.from_numpy(y_out)        # convertion from numpy to pytorch tensor
    dtype = torch.double
    device = torch.device("cpu")

    # Randomly initialize weights
    w1 = torch.randn(D_in, H, device=device, dtype=dtype)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype)
        # w1 = torch.empty(D_in, H, device=device, dtype=dtype).fill_(1)
        # w2 = torch.empty(H, D_out, device=device, dtype=dtype).fill_(1)
    x = torch.empty(sample_size,ntenors, device=device, dtype=dtype)
    y = torch.empty(ntenors, device=device, dtype=dtype)
    # x is input data for n-th sample
    # y is output data for n-th sample

    errors  = []
    iteration = []  # iteration arg for plotting
    u=1 # iteration counter

    # iteratiion loop
    for t in range(M):
    # Forward pass: compute predicted y
        loss =  0       # error averaging over all samples
        grad_w1 =  0    # gradient w1 averaging over all samples
        grad_w2 =  0    # gradient w2 averaging over all samples
        for k in range(n_samples):
            x[:,:]= xn[k,:,:]       # sample input extraction from batch
            y[:]= yn[k,:]           # sample output extraction from batch
            h = x.mm(w1)            # hidden layer input
            h_relu = h.clamp(min=0) # hidden layer output
            y_pred = h_relu.mm(w2)  # output layer input

            # Accumulate loss with averaging
            loss += (y_pred - y).pow(2).sum().item()/n_samples

            # Backprop to compute gradients of w1 and w2 with respect to loss
            # with averaging over all samples
            grad_y_pred = 2.0 * (y_pred - y)
            grad_w2 += h_relu.t().mm(grad_y_pred)
            grad_h_relu = grad_y_pred.mm(w2.t())
            grad_h = grad_h_relu.clone()
            grad_h[h < 0] = 0
            grad_w1 += x.t().mm(grad_h)
            # end of loop over samples

        # Updating weights using gradient descent
        w1 -= learning_rate * grad_w1 / n_samples
        w2 -= learning_rate * grad_w2 / n_samples
        # collecting values for illustration chart
        if chart_type == "loglog":
            iteration.append(math.log10(u))
            errors.append(math.log10(loss))
            labx = "log iterations"
            laby = "log  errors"
        if chart_type == "logy":
            iteration.append(u)
            errors.append(math.log10(loss))
            labx = "iterations"
            laby = "log  errors"
        if chart_type == "":
            iteration.append(u)
            errors.append(loss)
            labx = "iterations"
            laby = "errors"
        if chart_type == "logx":
            iteration.append(math.log10(u))
            errors.append(loss)
            labx="log iteration"
            laby ="errors"
        if t == 0:
            print("Start error = ", loss)
        u=u+1
        # end of the iteration loop
    span = time.time() - start
    span = int( span * 1000)*1.0/1000.
    print("Timing: " ,span," seconds")
    print("Final error = ",loss)
    numpy.savetxt("errors.csv", numpy.asarray(errors), delimiter=",")
    plt.plot(iteration,errors,linestyle="-",marker="")
    plt.title("Error vs iteration")
    plt.xlabel(labx)
    plt.ylabel(laby)
    plt.grid(which="both")
    plt.savefig('errors.png')
    plt.show()

ex1()
# initial example
"""
PyTorch: Tensors
----------------

A fully-connected ReLU network with one hidden layer and no biases, trained to
predict y from x by minimizing squared Euclidean distance.

This implementation uses PyTorch tensors to manually compute the forward pass,
loss, and backward pass.

A PyTorch Tensor is basically the same as a numpy array: it does not know
anything about deep learning or computational graphs or gradients, and is just
a generic n-dimensional array to be used for arbitrary numeric computation.

The biggest difference between a numpy array and a PyTorch Tensor is that
a PyTorch Tensor can run on either CPU or GPU. To run operations on the GPU,
just cast the Tensor to a cuda datatype.
"""
# import torch
#
#
# dtype = torch.float
# device = torch.device("cpu")
# # device = torch.device("cuda:0") # Uncomment this to run on GPU
#
# # N is batch size; D_in is input dimension;
# # H is hidden dimension; D_out is output dimension.
# N, D_in, H, D_out = 64, 1000, 100, 10
#
# # Create random input and output data
# x = torch.randn(N, D_in, device=device, dtype=dtype)
# y = torch.randn(N, D_out, device=device, dtype=dtype)
#
# # Randomly initialize weights
# w1 = torch.randn(D_in, H, device=device, dtype=dtype)
# w2 = torch.randn(H, D_out, device=device, dtype=dtype)
#
# learning_rate = 1e-6
# for t in range(500):
#     # Forward pass: compute predicted y
#     h = x.mm(w1)
#     h_relu = h.clamp(min=0)
#     y_pred = h_relu.mm(w2)
#
#     # Compute and print loss
#     loss = (y_pred - y).pow(2).sum().item()
#     print(t, loss)
#
#     # Backprop to compute gradients of w1 and w2 with respect to loss
#     grad_y_pred = 2.0 * (y_pred - y)
#     grad_w2 = h_relu.t().mm(grad_y_pred)
#     grad_h_relu = grad_y_pred.mm(w2.t())
#     grad_h = grad_h_relu.clone()
#     grad_h[h < 0] = 0
#     grad_w1 = x.t().mm(grad_h)
#
#     # Update weights using gradient descent
#     w1 -= learning_rate * grad_w1
#     w2 -= learning_rate * grad_w2

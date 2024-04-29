# Code in file tensor/two_layer_net_numpy.py
import numpy as np
import time

t1 = time.time()
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

lr = 1e-6
T = 500
for t in range(T):
    # forward pass
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # compute & print loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # backdrop and compute gradient of w1 w2
    grad_y_pred = 2 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    w1 += -lr * grad_w1
    w2 += -lr * grad_w2

print(N, ':', time.time() - t1)

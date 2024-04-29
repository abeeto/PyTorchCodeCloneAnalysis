import numpy as np

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# initialization
# Because optimal solution is (w1:2, w2:0, b:0),
# to initialize similar to above converges fast
w1 = 0
w2 = 0
b = 0
alpha = 0.01

def forward(x, w1, w2, b):
    y_hat = x*x*w2 + x*w1 + b
    return y_hat

def loss(y, y_hat):
    return ((y - y_hat) * (y - y_hat))

def gradient_w1(x, y, w1, w2, b):
    return (2 * x * (forward(x, w1, w2, b) - y))

def gradient_w2(x, y, w1, w2, b):
    return (2 * x * x * (forward(x, w1, w2, b) - y))

def gradient_b(x, y, w1, w2, b):
    return (2 * (forward(x, w1, w2, b) - y))

for epoch in range(1000):
    stop = 1
    print("\nepoch: ", epoch)
    for x_val, y_val in zip(x_data, y_data):
        # update weights simultaneously by using temporary variables
        w1_tmp = w1 - alpha * gradient_w1(x_val, y_val, w1, w2, b)
        w2_tmp = w2 - alpha * gradient_w2(x_val, y_val, w1, w2, b)
        b_tmp = b - alpha * gradient_b(x_val, y_val, w1, w2, b)
        w1 = w1_tmp
        w2 = w2_tmp
        b = b_tmp
        l = loss(y_val, forward(x_val, w1, w2, b))
        stop = l
        print("w1: %.2f, w2: %.2f, b: %.2f ====> loss: %.3f" % (w1, w2, b, l))
    print("prediction of 4 hours: %.2f" % forward(4, w1, w2, b))
    # early stopping
    if (stop < 0.00001):
        break
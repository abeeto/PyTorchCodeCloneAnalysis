# Gradient Descent essentially finding w that minimizes the loss (min point of graph)
# This gradient descent algorithm compute a random point gradient, i.e if gradient is +ve, move left.
# use the formula w = w - α (∂loss / ∂w ). to find global minimum.
# w = w - α ( gradient <<< which is 2x(xw - y) )

import matplotlib.pyplot as plt

x_data = [1,2,3]
y_data = [2,4,6]

# Random point for gradient testing
w = 1.0

# forward linear function [ y = x * w ] bias is removed
def forward(x):
    return x * w

# loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2

# Gradient
def gradient(x,y):
    return 2*x*(x*w-y)

# Before Train
print("Prediction Before Training", 4, forward(4))

# Training model, train 10 times
for i in range(10):
    for each_x , each_y in zip(x_data,y_data):
        grad = gradient(each_x,each_y)
        w = w - 0.01 * grad
        print("\tGradient: ", each_x, each_y, round(grad,2))
        l = loss(each_x,each_y)
    print("Training Number:",i, "/10 w = ", round(w,2), "Loss Error = ", round(l,2))

# After Train
print("Prediction After Training: ", forward(4))

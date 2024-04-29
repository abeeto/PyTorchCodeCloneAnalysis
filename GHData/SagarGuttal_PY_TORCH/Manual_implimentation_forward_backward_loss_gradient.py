## Problem_1 :
from array import array
import numpy as np

# f = w * x
# f = 2 * x

X = np.array([1,2,3,4,5,6], dtype=np.float32)
Y = np.array([2,4,6,8,10,12], dtype=np.float32)

w = 0.0

# model prdiction
def forward(x):
    return w * x

#Loss
# loss --> MSE

def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

#Gradient
# MSE = 1/N * (w*x - y)**2
# dj/dw = 1/N 2x (w*x - y)
def gradient(x,y,y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print(f"Prediction befor training: f(5) {forward(5):.3f}")

#training 
learning_rate = 0.01
n_iter = 12

for epoch in range(n_iter):
    # Prediction - forwardpass
    y_pred = forward(X)

    #loss 
    l = loss(Y, y_pred)

    #gradients 
    dw =  gradient(X,Y, y_pred)

    #updating wieght
    w -= learning_rate*dw

    if True:
        print(f'epoch {epoch+1}: w={w:.3f}, loss={l:.8f}')

print(f"Predictin after the traing : f(5) = {forward(9):.3f}")

## Problem 2

# f = 3 * x + 2 

def datas():
    y_list = []
    X = np.array([10,11,12,13,14,15,16], dtype=np.float64)
    for i in X:
        m = 3 * i + 2
        y_list.append(m)
    Y =  np.array(y_list, dtype=np.float32)

    return X, Y

w = 0

#prediction
def forward_pass():
    return w * X

#loss -- > MSE
def loss(y, y_pred):
    return ((y_pred-y)**2).mean()

#gradient
# MSE =  1/N * ((w * x + 2) - y)**2
# dj/dw = 1/N 2x*(w*x +2 - y)
def gradient(x,y,y_pred):
    return np.dot(2*x, y_pred-y).mean()

lr = 0.01

for epoch in range(100):
    y_pred = forward(X)
    L = loss(Y, y_pred)
    dw = gradient(X,Y, y_pred)
    w = w - lr*dw

    if True:
        print(f"epoch {epoch+1}: w={w:.3f}, loss={L:.38f}")
        
print(f"Prediction after the training : f(15) = {forward(15):.3f} ")
print(f"actual value {3 * 15 + 2}")

# # NN_2

# import numpyas np 

# class NeuralNet():
#     def __init__(self):
#         np.rndom.seed(1)

#         self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
#     def segmoid(self, x):
#         return 1 / (1+ np.exp(-x))
#     def segmoid_derivative(self, x):
#         return x * (1- x)

#     def train(self, traing_inputs, traning_outputs, traning_iteration):

#         output = self.think(traing_inputs)
#         error = traning_outputs - output
#         adjustemts  = np

import numpy as np 
# f = w * x

# f = 2 * x
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0 

# calculate model predctions 
def forward(X):
    return w * X
# loss = MSE
def loss(Y, y_predicted):
    return ((y_predicted-Y)**2).mean()

# gradient 
# MSE =  1/N * (w*x -y)**2
# dJ/dw = 1/N 2x (w*x - y)

def gradient(X, Y, y_predicted):
    return np.dot(2*X, y_predicted-Y).mean()

print(f'predction before traning: f(5) = {forward(5):.3f}')


# traning 
lr = 0.01
n_itrs = 10

for epoch in range(n_itrs):

    # predctions  = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients 

    dw = gradient(X,Y,y_pred)

    # update weight
    w -= lr *dw 

    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w  = {w:.3f}, loss = {l:.3f}')

print(f'predction after traning: f(5) = {forward(5):.3f}')
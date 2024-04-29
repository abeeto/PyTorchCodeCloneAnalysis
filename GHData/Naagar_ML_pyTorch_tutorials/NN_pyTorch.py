# 1. Design model (input, output siez, forward pass)
# 2. construcht loss and optimizer 
# 3. traning loop 
#       - forward pass
#       - backward pass
#       - updates the weights 

import torch.nn as nn

import numpy as np 
import torch 
# f = w * x

# f = 2 * x
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape

print(n_samples, n_samples)
# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) 

input_size = n_features
output_size = n_features

# calculate model predctions 
# model = nn.Linear(input_size, output_size)

class LinearModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()

        # define layers
        self.lin = nn.Linear(input_size, output_size) 

    def forward(self, x):
        return self.lin(x)

model = LinearModel(input_size, output_size)
# def forward(X):
#     return w * X
# loss = MSE
# def loss(Y, y_predicted):
#     return ((y_predicted-Y)**2).mean()

# gradient 
# MSE =  1/N * (w*x -y)**2
# dJ/dw = 1/N 2x (w*x - y)

# def gradient(X, Y, y_predicted):
#     return np.dot(2*X, y_predicted-Y).mean()

print(f'predction before traning: f(5) = {model(X_test).item():.3f}')


# traning 
lr = 0.001
n_itrs = 1000

loss  = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr )

for epoch in range(n_itrs):

    # predctions  = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients  = backward_pass

    # dw = gradient(X,Y,y_pred)
    l.backward()  #  dl/dw

    # # update weight
    optimizer.step()

    # zero gradients 
    optimizer.zero_grad()
    # with torch.no_grad():
        
    #      w -= lr * w.grad 

    if epoch % 200 == 0:
        [w,b] = model.parameters()
        print()
        print(f'epoch {epoch+1}: w  = {w[0].item():.3f}, loss = {l:.3f}')

print(f'predction after traning: f(5) = {model(X_test).item():.3f}')
# 1) Design model (input, output size, forward pass)
# 2) Construct Loss and Optimizer
# 3) Training Loop
# - forward pass : compute prediction
# - backward pass : gradients
# - update weights
import torch
import torch.nn as nn
import numpy as np
import os

os.system('cls')

X = torch.tensor([[1], [2], [3], [4]], dtype = torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype = torch.float32)
X_test = torch.tensor([5], dtype = torch.float32)

# w = torch.tensor(0.0, dtype = torch.float32, requires_grad=True)


n_samples, n_features = X.shape
print(n_samples, n_features)
intput_size = n_features
output_size = n_features

class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()

        #! def layers :
        self.lin = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.lin(x)

# model = nn.Linear(intput_size, output_size)
model = LinearRegression(intput_size, output_size)

#! model prediction
# def forward(x):
#     return w * x

#! gradients
#! MSE : 1/N * (w*X - y)**2
#! dJ/dW = 1/N 2x (w*x - y)
# TODO : dJ/dW = dJ/dy * dy/dW = 1/N * 2 * X * (w*x - y)
# J = 1/N * (w*X - y_pred)**2 => dJ = 1/N * 2 * (w*x - y)
# y = w * x => dy/dW = x
def gradient(x, y, y_pred):
    return np.dot(2*x, y_pred-y).mean()

print(f"prediction before training : {model(X_test).item():.3f}")

#! training:
learning_rate = 0.01
n_iters = 1000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

for epoch in range(n_iters):
    #! prediction = forward pass
    y_pred = model(X)

    #! loss :
    loss_ = loss(Y, y_pred)

    #! gradients:
    #! dw = gradient(X, Y, y_pred)    
    loss_.backward()

    #! update weight:    
    # with torch.no_grad():
    #     w -= learning_rate * w.grad
    optimizer.step()
    # w.grad.zero_()
    optimizer.zero_grad()

    if epoch % 2 == 1:
        [w, b] = model.parameters()
        print(f"epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {loss_:.8f}")
print(f'predition after training : f(5) = {model(X_test).item():.3f}')
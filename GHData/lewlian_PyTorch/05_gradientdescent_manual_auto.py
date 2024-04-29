# Prediction: PyTorch MOdel
# Gradients Computation: Autograd
# Loss Computation: PyTorch Loss
# Parameter updates : Pytorch Optimizer

# Training pipeline usually in 3 steps
# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Perform Training loop:
#   - forward pass: compute prediction
#   - backward pass: get gradients (can be done by pyTorch)
#   - update weights

import torch
import torch.nn as nn  # neural network module

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32) # 1-dimension with 1 feature 
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()

        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size,output_size)


# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# calculate model prediction
# def forward(x):
#     return w * x

# calculate loss = MSE in the case of linear regression
# def loss(y, y_predicted):
#     return ((y_predicted-y)**2).mean()

print(f'Prediction before trianing: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
# Stochastic Gradient Descent
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradient = backward pass
    l.backward()  # calculates dl/dx

    # update weights ( go in the negative direction of gradient)
    # with torch.no_grad():
    #     w -= learning_rate * w.grad
    optimizer.step()

    # zero gradients (everytime we call backward, it will accumulate the gradient in w.grad)
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after trianing: f(5) = {model(X_test).item():.3f}')

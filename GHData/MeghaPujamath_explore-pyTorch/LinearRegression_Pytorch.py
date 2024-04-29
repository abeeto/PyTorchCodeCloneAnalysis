# 1. Design model
# 2. Calculate Loss and Optimizer
# 3. Training Loop
#       a. forward pass: compute prediction
#       b. backward pass: gradients
#       c. update weights

from numpy import dtype
from sympy import N
import torch
import torch.nn as nn

# Linear model
# y = w * x
# y = 2 * x


class LinearRegression(nn.Module):

    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()

        # define layers
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


X = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8], [10]], dtype=torch.float32)

X_test = torch.tensor([20], dtype=torch.float32)

# 1. define linear model
n_samp, n_featu = X.shape
input_size = n_featu
output_size = n_featu

model = LinearRegression(input_size, output_size)

learning_rate = 0.01
num_epochs = 100

# 2. calculate loss = Mean Square Error
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 3. Training loop
for epoch in range(num_epochs):
    # 1. forward pass and loss calculation
    y_pred = model.forward(X)

    l = loss(Y, y_pred)

    # 2. gradient calculation - backward pass
    l.backward()

    # 3. update weights
    optimizer.step()

    # 4. empty gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f"epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}")

print(f"Prediction after training {model(X_test).item():.4f}")

import torch
import torch.nn as nn
# 1 design model (input, otput, forwardpass)
# 2 Construct the loss and optimizer
# 3 Training loop
# - forward pass: compute prediction
# - backprop: gradients
# - update weights

# f = w * x

# f = 2 * x
# 2d because the number of rows has to be the number of outputs
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

nsamples, nfeatures = X.shape
print(nsamples, nfeatures)
inputSize = nfeatures
outputSize = nfeatures
#model = nn.Linear(inputSize, outputSize)

class LinearRegression(nn.Module):
    def __init__(self, inputDim, outputDim):
        super().__init__()
        self.lin = nn.Linear(inputDim, outputDim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(inputSize, outputSize)

print(f'prediction before training; f(5) = {model(X_test).item():.3f}')

learning_rate = 0.01
n_iters = 100

# Mean swuared error
loss = nn.MSELoss()
# Optimizer stochastic gradient descent, for the gradient
# model.parameters is the weights
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward prop
    l.backward() # calculate the gradient dl/dw

    #update weigths
    optimizer.step()

    if epoch % 10 == 0:
        [w,b] = model.parameters()
        print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss {l:.8f}')

    # Must empty gradients
    optimizer.zero_grad()

print(f'prediction after training; f(5) = {model(X_test).item():.3f}')

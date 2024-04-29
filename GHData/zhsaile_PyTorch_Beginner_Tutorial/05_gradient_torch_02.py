# 1) Design model (input, output size, forward pass)

# 2) Construct loss and optimizer

# 3) Training loop
#  - forward pass: compute prediction
#  - backward pass: gradient
#  - update weights
import torch
import torch.nn as nn

# f = w * x

# f = 2 * x
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)
x_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)
print(f'Prediction before training: f(5)={model(x_test).item():.3f}')

#w = torch.tensor(0., dtype=torch.float32, requires_grad=True)

# model prediction
#def forward(x):
    #return w * x

# loss and optimizer
lr = 0.01
loss = nn.MSELoss()
optimizer = torch.optim.SGD([model.parameters()], lr=lr)

# gradient
# MSE = 1/N * (w*x-y)**2
# dL/dw = 1/N * 2x * (w*x-y)

def gradient(x, y, y_pred):
    return np.dot(2*x, y_pred-y).mean()


# Training
epochs = 100

for epoch in range(epochs):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward
    l.backward() # dl/dw

    # update weights
    #with torch.no_grad():
        #w -= lr*w.grad
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()
    #w.grad.zero_()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0]:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5)={model(x_test).item():.3f}')

x = torch.tensor([1., 2., 3.], requires_grad=True)
w = torch.tensor(3., requires_grad=True)

y = (w*x).mean()
y.backward()
w.grad.zero_()
y = (w*x).mean()
y.backward()
print(w.grad)

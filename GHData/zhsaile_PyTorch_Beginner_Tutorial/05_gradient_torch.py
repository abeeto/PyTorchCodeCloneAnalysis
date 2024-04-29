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
X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0., dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w * x

def loss(y, y_pred):
    return ((y-y_pred)**2).mean()

# gradient
# MSE = 1/N * (w*x-y)**2
# dL/dw = 1/N * 2x * (w*x-y)

def gradient(x, y, y_pred):
    return np.dot(2*x, y_pred-y).mean()

print(f'Prediction before training: f(5)={forward(5):.3f}')

# Training
lr = 0.01
epochs = 100

for epoch in range(epochs):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward
    l.backward() # dl/dw

    # update weights
    with torch.no_grad():
        w -= lr*w.grad

    # zero gradients
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

#print(f'Prediction after training: f(5)={forward(5):.3f}')

x = torch.tensor([1., 2., 3.], requires_grad=True)
w = torch.tensor(3., requires_grad=True)

y = (w*x).mean()
y.backward()
w.grad.zero_()
y = (w*x).mean()
y.backward()
print(w.grad)

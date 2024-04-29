import numpy as np

# f = w * x

# f = 2 * x
X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)

w = 0.

# model prediction
def forward(x):
    return w * x

# loss
def loss(y, y_pred):
    return ((y-y_pred)**2).mean()

# gradient
# MSE = 1/N * (w*x-y)**2
# dL/dw = 1/N * 2x * (w*x-y)

def gradient(x, y, y_pred):
#    return np.dot(2*x, y_pred-y)/x.size
    print(np.dot(2*x, y_pred-y)/x.size)
    return np.dot(2*x, y_pred-y).mean()

print(f'Prediction before training: f(5)={forward(5):.3f}')

# Training
lr = 0.01
epochs = 10

for epoch in range(epochs):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients
    dw = gradient(X, Y, y_pred)

    # update weights
    w -= lr*dw

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5)={forward(5):.3f}')

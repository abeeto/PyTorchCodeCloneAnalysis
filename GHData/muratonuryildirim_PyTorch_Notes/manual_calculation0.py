import numpy as np

# function: f=2x
x = np.array([1, 2, 3, 4], dtype=np.float32)
y = np.array([2, 4, 6, 8], dtype=np.float32)
w = 0.0


def forward(x):
    return w * x


# loss : MSE
def loss(y, y_pred):
    return ((y_pred-y)**2).mean()


# gradient: 1/N * (2 * x (w * x - y))
def gradient(x, y, y_pred):
    return np.dot(2*x, y_pred-y).mean()


print(f'Prediction before training for f(5) {forward(5)}')

# training
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    # forward pass
    y_pred = forward(x)
    # loss
    l = loss(y, y_pred)
    # backward pass
    dw = gradient(x, y, y_pred)
    # update weight(s)
    w -= learning_rate * dw

    print(f'epoch {epoch} : w = {w}, loss= {l}')
    print(f'Prediction after training for f(5) {forward(5)}')
    
    

import numpy as np

# Linear model
# y = w * x
# y = 2 * x

X = np.array([1, 2, 3, 4, 5], dtype=np.float32)
Y = np.array([2, 4, 6, 8, 10], dtype=np.float32)

w = 0.0

# 1. define linear model


def model(x):
    return w * x

# 2. calculate loss = Mean Square Error


def loss(y, y_pred):
    return ((y_pred - y)**2).mean()


print(f"Prediction before training {model(20):.4f}")

# 3. calculate gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N 2x(w*x - y)


def gradient(x, y, y_pred):
    return np.dot(2*x, (y_pred-y)).mean()


# 4. Training loop
learning_rate = 0.01
num_epochs = 10

for epoch in range(num_epochs):
    # 1. forward pass and loss calculation
    y_pred = model(X)

    l = loss(Y, y_pred=y_pred)

    # 2. gradient calculation
    dw = gradient(X, Y, y_pred)

    # 3. update weights
    w -= learning_rate * dw

    if epoch % 1 == 0:
        print(f"epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}")

print(f"Prediction after training {model(20):.4f}")

import numpy as np
import math
import matplotlib.pyplot as plt

# Create random input and output data
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)
plt.plot(y)

# Randomly initiative weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()
e = np.random.randn()
f = np.random.randn()

learning_rate = 1e-7
N = 100000
for t in range(N):
    # Forward pass: compute predicted y
    # y = a + b x + c x^2 + d x^3 + e x^4
    y_pred = a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(100 * t / N, loss)

    # Plot intermediate predictions
    if t % (N // 2) == 0:
        plt.plot(y_pred)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = (grad_y_pred).sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()
    grad_e = (grad_y_pred * x ** 4).sum()
    grad_f = (grad_y_pred * x ** 5).sum()

    # Update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d
    e -= learning_rate * grad_e
    f -= learning_rate * grad_f

print(f"Result: y = {a} + {b} x + {c} x^2 + d{d} x^3 + e{e} x^4 + f{f} x^5")

# Plot y and y predicted
plt.legend("y", "y pred")
plt.ylim([np.min(y), np.max(y)])
plt.show()

# Training Data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


# our model forward pass
def forward(x):
    return x * w

# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

# compute gradient
def gradient(x, y):
    return 2 * x * (x * w - y)


# Before training
print("predict (before training)", 4, forward(4))

# Training loop
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w - 0.01 * grad
        print("\tgrad: ", x_val, y_val, round(grad,2))

    l = loss(x_val, y_val)
    print("progress:", epoch, "w=", w, "loss=", l)
    print()

# After training
print("predict (after training)", "4 hours", forward(4))
import matplotlib.pyplot as plt

# our model forward pass
def forward(x):
    return x**2 * w2 + x * w1 + b

# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

# compute gradient
def gradient_w1(x, y):
    return 2 * x * (x**2 * w2 + x * w1 + b)

def gradient_w2(x, y):
    return 2 * x**2 * (x**2 * w2 + x * w1 + b)

def gradient_b(x, y):
    return 2 * (x**2 * w2 + x * w1 + b)



x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = 2.0
w2 = 0.0
b = 0.0

# Before training
print("predict (before training)", 4, forward(4))


w1_list = []
w2_list = []
b_list = []
mse_list = []
# Training loop
for epoch in range(1000):
    for x_val, y_val in zip(x_data, y_data):
        grad_w1 = gradient_w1(x_val, y_val)
        grad_w2 = gradient_w2(x_val, y_val)
        grad_b = gradient_b(x_val, y_val)
        w1 = w1 - 0.00001 * grad_w1
        w2 = w2 - 0.00001 * grad_w2
        b = b - 0.00001 * b
        print("\tgrad: ", x_val, y_val, grad_w1, grad_w2, grad_b)

    l = loss(x_val, y_val)
    print("progress: {}, w1={}, w2={}, b={}, loss={}".format(epoch,w1,w2,b,l))
    print()
    w1_list.append(w1)
    w2_list.append(w2)
    b_list.append(b)
    mse_list.append(l)

# After training
print("predict (after training)", "4 hours", forward(4))


plt.plot(w1_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w1')
plt.show()
plt.close()

plt.plot(w2_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w2')
plt.show()
plt.close()

plt.plot(b_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('b')
plt.show()
plt.close()

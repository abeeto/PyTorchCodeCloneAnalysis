import matplotlib.pyplot as plt


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = 1.0
epochs = 100
lr = 0.01


def forward(x):
    return x * w


def loss_sgd(x, y):
    y_hat = forward(x)
    return (y - y_hat) ** 2


def sgd(x, y):
    gradient = 2 * x * (x * w - y)
    return gradient


def train():
    global w, epochs, lr
    epoch_list = []
    loss_list = []
    loss = 0
    for epoch in range(epochs):
        for x, y in zip(x_data, y_data):
            gradient = sgd(x, y)
            w -= (lr * gradient)
            loss = loss_sgd(x, y)

        epoch_list.append(epoch)
        loss_list.append(loss)
        print('{:>3}/{}, w={:.3}, loss={:.3}'.format(epoch+1, epochs, w, loss))

    plt.plot(epoch_list, loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


if __name__ == "__main__":
    train()

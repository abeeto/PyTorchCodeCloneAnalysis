import matplotlib.pyplot as plt


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = 1.0
epochs = 100
lr = 0.01


def forward(x):
    return x * w


def gradient_gd(x_data, y_data):
    gradient = 0
    for x, y in zip(x_data, y_data):
        gradient += 2 * x * (x * w - y)
    return gradient / len(x_data)


def cost_gd(x_data, y_data):
    cost = 0
    for x, y in zip(x_data, y_data):
        cost += (forward(x) - y) ** 2
    return cost / len(x_data)


def train():
    global w, epochs, lr
    epoch_list = []
    cost_list = []
    for epoch in range(epochs):
        cost = cost_gd(x_data, y_data)
        gradient = gradient_gd(x_data, y_data)
        w -= (lr * gradient)

        epoch_list.append(epoch)
        cost_list.append(cost)
        print("{:>3d}/{}, w:{:.3}, cost:{:.3}".format(epoch + 1, epochs, w, cost))

    plt.plot(epoch_list, cost_list)
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.show()


def infer(x_infer):
    print("infer:x={:4.2}, y_predict={:4.2}".format(x_infer, forward(x_infer)))


if __name__ == '__main__':
    train()

import numpy as np
import matplotlib.pyplot as plt
import torch

print(torch.cuda.is_available())

def visualize_relation(x, y):
    """
    For visualising the relation b/w the
    Explanatory variable and the Target Variable
    :return:
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c="green", s=250, label="Original Data")
    plt.show()


def np_to_torch(x, y):
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    return x, y


def train_model(x, y):
    input_size = 1
    hidden_size = 1
    output_size = 1
    learning_rate = 0.001

    x_train, y_train = np_to_torch(x, y)
    w1 = torch.rand(input_size, hidden_size, requires_grad=True)
    b = torch.rand(hidden_size, output_size, requires_grad=True)

    for iter in range(1, 7001):
        y_pred = x_train * w1 + b
        loss = (y_pred - y_train).pow(2).sum()
        if iter % 100 == 0:
            print(f"Iter: {iter}, Loss: {loss}")
        loss.backward()

        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            b -= learning_rate * b.grad
            w1.grad.zero_()
            b.grad.zero_()
    print(loss, w1, b)
    return w1, b


def predict_nn(x, weight, bias):
    
    print("Predictions ..")
    return x.mm(weight).add(bias).detach().numpy()

if __name__ == "__main__":

    x_train = np.array(
                [
                  [4.7], [2.4], [7.5], [7.1], [4.3],
                  [7.8], [8.9], [5.2], [4.59], [2.1],
                  [8], [5], [7.5], [5], [4],
                  [8], [5.2], [4.9], [3], [4.7],
                  [4], [4.8], [3.5], [2.1], [4.1]
                ],
                dtype=np.float32
                )

    y_train = np.array(
                [
                  [2.7], [1.6], [3.09], [2.4], [2.4],
                  [3.3], [2.6], [1.96], [3.13], [1.76],
                  [3.2], [2.1], [1.6], [2.5], [2.2],
                  [2.75], [2.4], [1.8], [1], [2],
                  [1.6], [2.4], [2.6], [1.5], [3.1]
                ],
                dtype=np.float32
    )

    # train model
    w, b = train_model(x_train, y_train)

    #predict
    x_test = torch.tensor([
                            [2.6]]
                         )
    y_pred = predict_nn(x_test, w, b)
    print(x_test, y_pred)

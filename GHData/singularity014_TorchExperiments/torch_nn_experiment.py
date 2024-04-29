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
    x, y = np_to_torch(x, y)
    learning_rate = 1e-4

    input_size = 1
    hidden_size = 5
    output_size = 1

    model = torch.nn.Sequential(
                                torch.nn.Linear(input_size, hidden_size),
                                torch.nn.ReLU(),
                                torch.nn.Linear(hidden_size, output_size)
                                )
    for i in range(500):

        y_pred = model(x)

        loss_fn = torch.nn.MSELoss(reduction="sum")
        loss = loss_fn(y_pred, y)
        print(i, loss.item())

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

    return model


def predict_nn(model, x):
    print("Predictions ..")
    return model(x).detach().numpy()

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
    trained_model = train_model(x_train, y_train)
    print(trained_model)

    x_test = torch.from_numpy(x_train)
    y_pred = predict_nn(trained_model, x_test)
    print(x_test)

    plt.scatter(x_train, y_train)
    plt.plot(x_train, y_pred)
    plt.show()
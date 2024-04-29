import torch
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class NeuralNet(torch.nn.Module):
    """
    Artificial Neural Network
    ____Version 1.0____
    """

    def __init__(self, params, activation_functions):
        """
        :param params: (list) specifying the dimension of each layers,
        shape=[n_features, # of nodes in each layer..., n_classes]
        :param activation_functions: a list object specifying the activation functions,
        length = len(params-2)
        """
        super(NeuralNet, self).__init__()
        self.params = params
        self.activation_functions = []
        self.layers = []

        for i, dims in enumerate(zip(params[:-1], params[1:])):
            setattr(self, 'Layer_{}'.format(i), torch.nn.Linear(dims[0], dims[1]))
            self.layers.append(eval('self.Layer_{}'.format(i)))

        for i, act_func in enumerate(activation_functions):
            setattr(self, 'activation_func_{}'.format(i), eval('torch.nn.' + act_func))
            self.activation_functions.append(eval('self.activation_func_{}'.format(i)))

    def forward(self, x):
        """
        Backpropagation
        :param x: (torch tensor) shape=(n_samples, n_features)
        :return:
        """
        z = x
        for i in range(len(self.activation_functions)):
            z = self.layers[i](z)
            z = self.activation_functions[i]()(z)
        return self.layers[-1](z)

    def fit(self, x, y, iterations=1000, eta=1e-3, optimizer="Adam", print_loss=False):
        """
        :param x: (torch tensor) shape=(n_samples, n_features)
        :param y: torch tensor, shape=(n_samples,)
        :param iterations: int, number of iterations, default 1000
        :param eta: float, learning rate for gradient descent, default 1e-3
        :param optimizer: str, specify an optimizer for parameter updates, defaul Adam
        :param print_loss: boolean, if True, print loss and training accuracy, default False
        """
        optimizer = eval('torch.optim.' + optimizer)
        optimizer = optimizer(self.parameters(), lr=eta)
        criterion = torch.nn.CrossEntropyLoss()

        for i in range(iterations):

            optimizer.zero_grad()

            y_hat = self.forward(x)

            loss = criterion(y_hat, y)

            if print_loss:
                pred = self.predict(x)
                accuracy = torch.sum(pred == y).numpy() / y.size()[0]
                print('Iteration {} || loss: {:0.4f}    || accuracy: {:0.3f}'.format(i, loss.item(), accuracy))

            loss.backward()

            optimizer.step()

    def predict(self, x):
        """
        Make predictions
        :param x: (torch tensor) shape=(n_samples, n_features)
        :return: (torch tensor) shape=(n_samples,)
        """
        z = x
        for i in range(len(self.activation_functions)):
            z = self.layers[i](z)
            z = self.activation_functions[i]()(z)

        return torch.argmax(self.layers[-1](z), dim=1)


def main():
    mnist_trainset = datasets.MNIST(root='./data', train=True,
                                    download=True, transform=transforms.ToTensor())
    mnist_testset = datasets.MNIST(root='./data', train=False,
                                   download=True, transform=transforms.ToTensor())

    x = mnist_trainset.data.reshape(-1, 28 * 28).float()
    y = mnist_trainset.targets

    model = NeuralNet([784, 64, 32, 10], ['ReLU', 'Sigmoid'])

    model.fit(x, y, iterations=50, optimizer='Adam', print_loss=True)


if __name__ == '__main__':
    main()

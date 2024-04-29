import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class LinearRegression():

    def __init__(self):

        self.weights = None
        self.bias = None

    def fit(self, X, y):

        num_epochs = 10000
        N,K = X.shape
        inputs = Variable(X)
        actual = Variable(y)

        criterion = nn.MSELoss()
        linear = torch.nn.Linear(K, 1, bias=True)
        optimizer = optim.Adam(linear.parameters())

        for epoch in range(num_epochs):

            outputs = linear(inputs)
            loss = criterion(outputs, actual)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(linear.weight)

        self.weight = linear.weight.data.numpy()[0]
        self.bias = linear.bias.data.numpy()[0]


    def predict(self, X):

        return self.bias + X.dot(self.weight)



if __name__ == "__main__":

    N = 500
    K = 5

    mean = np.random.normal(5, .2, size = K)

    X = np.random.multivariate_normal(mean = mean, cov = np.identity(K), size = (N))
    B = np.random.uniform( size = K)
    y = X.dot(B).reshape(N,1)# + np.random.normal(0,1, size = N)

    X_torch = torch.from_numpy(X).float()
    y_torch = torch.from_numpy(y).float()

    Linear = LinearRegression()
    Linear.fit(X_torch, y_torch)

    print("actual Beta", B)

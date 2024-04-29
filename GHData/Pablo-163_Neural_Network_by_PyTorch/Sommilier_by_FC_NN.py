import torch
import random as rd

# to fix results for each training
rd.seed(0)

# get dataset from sklearn - dataset about wines
import sklearn.datasets

wines = sklearn.datasets.load_wine()

# print(wines.data.shape)

# split to train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(wines.data[:, :2],
                                                    wines.target,
                                                    test_size=0.2,  # proportion of test
                                                    shuffle=True)

# make data useful for torch
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


# NN architecture

class Sommelier(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(Sommelier, self).__init__()
        # use 3 fully connected layers with SoftMax on the output
        # number of hidden neurons can be easily changed
        self.fc1 = torch.nn.Linear(2, n_hidden_neurons)
        self.act_fn1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.act_fn2 = torch.nn.Sigmoid()
        # 3 neurons on output for each wine's class
        self.fc3 = torch.nn.Linear(n_hidden_neurons, 3)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn1(x)
        x = self.fc2(x)
        x = self.act_fn2(x)
        x = self.fc3(x)
        return x

    # separate SoftMax for easily compute cross-entropy
    def inference(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x


wine_sommelier = Sommelier(6)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(wine_sommelier.parameters(), lr=0.003)

# make batch training
import numpy as np

batch_size = 20

for epoch in range(1000):
    # shuffle data for each epoch
    order = np.random.permutation(len(X_train))
    for start in range(0, len(X_train), batch_size):
        optimizer.zero_grad()

        batch_indexes = order[start:start + batch_size]
        # make a batch of data
        x_b = X_train[batch_indexes]
        y_b = y_train[batch_indexes]

        pred = wine_sommelier.forward(x_b)

        loss = loss_fn(pred, y_b)
        loss.backward()

        optimizer.step()
    # check progress once a 50 epochs
    if epoch % 50 == 0:
        test_preds = wine_sommelier.forward(X_test)
        test_preds = test_preds.argmax(dim=1)
        print((test_preds == y_test).float().mean())

    # get accuracy around 0.7

# visualize some data

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 8)

n_classes = 3
plot_colors = ['g', 'orange', 'black']
plot_step = 0.02

x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

xx, yy =  torch.meshgrid(torch.arange(x_min, x_max, plot_step),
                         torch.arange(y_min, y_max, plot_step))

preds = wine_sommelier.inference(
    torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1))

preds_class = preds.data.numpy().argmax(axis=1)
preds_class = preds_class.reshape(xx.shape)
plt.contourf(xx, yy, preds_class, cmap='Accent')

for i, color in zip(range(n_classes), plot_colors):
    indexes = np.where(y_train == i)
    plt.scatter(X_train[indexes, 0],
                X_train[indexes, 1],
                c=color,
                label=wines.target_names[i],
                cmap='Accent')
    plt.xlabel(wines.feature_names[0])
    plt.ylabel(wines.feature_names[1])
    plt.legend()

plt.show()

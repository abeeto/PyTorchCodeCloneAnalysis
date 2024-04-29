import torch
import numpy as np

# dataset is taken from torch_vision
import torchvision.datasets

numbers_train = torchvision.datasets.MNIST('./', download=True, train=True)
numbers_test = torchvision.datasets.MNIST('./', download=True, train=False)

# separate into features and predicted value

x_train = numbers_train.train_data
y_train = numbers_train.train_labels

x_test = numbers_test.test_data
y_test = numbers_test.test_labels

# make train set useful
x_train = x_train.float()  # [60000, 28, 28]
#y_train = y_train.float()

x_test = x_test.float()
#y_test = y_test.float()

# make vector from image
x_train = x_train.reshape([-1, 28 * 28])
x_test = x_test.reshape([-1, 28 * 28])


class NumberNet(torch.nn.Module):
    def __init__(self, n_hidden):
        super(NumberNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, n_hidden)
        self.ac1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden, 10)  # output for 10 numbers

    def forward(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        x = self.fc2(x)
        return x


# let use 50 neurons
num_net = NumberNet(80)

# loss-func will be cross-entropy
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(num_net.parameters(), lr=0.001)

accuracy_his =[]
loss_his = []

# training by batch
batch_size = 100
for epoch in range(100):
    # generate random set of indexes
    order = np.random.permutation(len(x_train))
    for start_id in range(0, len(x_train), batch_size):
        optimizer.zero_grad()

        batch_ids = order[start_id:start_id + batch_size]

        x_batch = x_train[batch_ids]
        y_batch = y_train[batch_ids]

        preds = num_net.forward(x_batch)
        loss_val = loss(preds, y_batch)

        loss_val.backward()

        optimizer.step()
    test_preds = num_net(x_test)
    loss_his.append((loss(test_preds, y_test)).item())
    # test_preds.argmax(dim=1) give number witch max likelihood
    accuracy = (test_preds.argmax(dim=1) == y_test).float().mean() # bool -> float
    accuracy_his.append(accuracy)
    print(accuracy)


import matplotlib.pyplot as plt
plt.plot(loss_his)
plt.plot(accuracy_his)

plt.show()

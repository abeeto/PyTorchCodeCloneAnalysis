import random
import warnings

import torch
import numpy as np
import torchvision.datasets
import matplotlib.pyplot as plt
warnings.simplefilter('ignore')

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

MNIST_train = torchvision.datasets.MNIST('./', download=True, train=True)
MNIST_test = torchvision.datasets.MNIST('./', download=True, train=False)

X_train = MNIST_train.train_data
y_train = MNIST_train.train_labels
X_test = MNIST_test.test_data
y_test = MNIST_test.test_labels

print(X_train.dtype, y_train.dtype)

X_train = X_train.float()
X_test = X_test.float()

print(X_train.shape, X_test.shape)

print(y_train.shape, y_test.shape)

# plt.imshow(X_train[0, :, :])
# plt.show()
print(y_train[0])

X_train = X_train.reshape([-1, 28 * 28])
X_test = X_test.reshape([-1, 28 * 28])


class MNISTNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(MNISTNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, n_hidden_neurons)
        self.ac1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        x = self.fc2(x)
        return x


mnist_net = MNISTNet(100)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mnist_net.parameters(), lr=1.0e-3)

batch_size = 128

test_accuracy_history = []
train_accuracy_history = []
test_loss_history = []
train_loss_history = []

for epoch in range(300):
    order = np.random.permutation(len(X_train))

    for start_index in range(0, len(X_train), batch_size):
        optimizer.zero_grad()

        batch_indexes = order[start_index:start_index + batch_size]

        X_batch = X_train[batch_indexes]  # .to(device)
        y_batch = y_train[batch_indexes]  # .to(device)

        preds = mnist_net.forward(X_batch)

        loss_value = loss(preds, y_batch)
        loss_value.backward()

        optimizer.step()

    test_preds = mnist_net.forward(X_test)
    train_preds = mnist_net.forward(X_train)
    test_loss_history.append(loss(test_preds, y_test))
    train_loss_history.append(loss(train_preds, y_train))

    accuracy = (test_preds.argmax(dim=1) == y_test).float().mean()
    test_accuracy_history.append(accuracy)
    train_accuracy_history.append((train_preds.argmax(dim=1) == y_train).float().mean())
    print(accuracy)

fig, ax = plt.subplots(1, figsize=(8, 6))
ax.plot(test_accuracy_history, label='TEST')
ax.plot(train_accuracy_history, label='TRAIN')
plt.legend(loc="lower right", title="Legend Title")
plt.show()


fig, ax = plt.subplots(1, figsize=(8, 6))
ax.plot(test_loss_history, label='TEST')
ax.plot(train_loss_history, label='TRAIN')
plt.legend(loc="lower right", title="Legend Title")
plt.show()

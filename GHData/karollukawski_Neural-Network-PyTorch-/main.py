import itertools
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from torch import nn
from torch import optim
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Creating the dataset for classification

X, y = make_classification(n_samples=10000, n_features=4, n_classes=3, n_redundant=0, n_informative=3, n_clusters_per_class=2)

# Visualizing the dataset

plt.title("Multi-class data, 4 informative features, 3 classes", fontsize="large")
plt.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=25, edgecolor="k")

plt.show()

# Splitting the dataset into test and train sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=26)

# Visualizing the data

fig, (train_ax, test_ax) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10, 5))
train_ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Spectral)
train_ax.set_title("Training Data")
train_ax.set_xlabel("Feature #0")
train_ax.set_ylabel("Feature #1")

test_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
test_ax.set_xlabel("Feature #0")
test_ax.set_title("Testing data")
plt.show()

# Converting data to torch tensors

class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len

batch_size = 2

# Instantiating training and testing data
train_data = Data(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# Checking working

for batch, (X, y) in enumerate(train_dataloader):
    print(f"Batch: {batch+1}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    break

print("Batch complete")

# Defining the dimensions of the network

input_dim = 4
hidden_dim = 2
output_dim = 1

# Defining the neural network class

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.sigmoid(self.layer_2(x))

        return x

# Instantiating the classifier

model = NeuralNetwork(input_dim, hidden_dim, output_dim)
print(model)

# Training model

learning_rate = 0.1

loss_fn = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), learning_rate)

num_epochs = 100
loss_values = []

for epoch in range(num_epochs):
    for X, y in train_dataloader:
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(-1))
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()

print("Training Complete")

# Visualizing loss curve

step = np.linspace(0, 100, 335000)

fig, ax = plt.subplots(figsize=(8,5))
plt.plot(step, np.array(loss_values))
plt.title("Step-wise Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

y_pred = np.array(np.zeros)

total = 0
correct = 0

with torch.no_grad():
    for X, y in test_dataloader:
        outputs = model(X)
        predicted = np.where(outputs < 0.5, 0, 1)
        predicted = list(itertools.chain(*predicted))
        y_pred = np.append(predicted, 3)
        y_test = np.append(y, 3)
        total += y.size(0)
        correct += (predicted == y.numpy()).sum().item()

print(f'Accuracy of the network on the 3300 test instances: {100 * correct // total}%')

# Showing performing of our model - confusion matrix

cf_matrix = confusion_matrix(y_test, y_pred)

plt.subplots(figsize=(8, 5))

sns.heatmap(cf_matrix, annot=True, cbar=False, fmt="g")

plt.show()


############################
#Additional functions to use
############################

#Sigmoid function
#
# x = np.linspace(-5, 5, 50)
# z = 1/(1 + np.exp(-x))
#
# plt.subplots(figsize=(8, 5))
# plt.plot(x, z)
# plt.grid()
# plt.show()
#
#Tanh function (hyperbolic tangent)
#
# x = np.linspace(-5, 5, 50)
# z = np.tanh(x)
#
# plt.subplots(figsize=(8, 5))
# plt.plot(x, z)
# plt.grid()
# plt.show()
#
#Rectified Linear Unit (ReLU)
#
# x = np.linspace(-5, 5, 50)
# z = [max(0, i) for i in x]
#
#
# plt.subplots(figsize=(8, 5))
# plt.plot(x, z)
# plt.grid()
# plt.show()
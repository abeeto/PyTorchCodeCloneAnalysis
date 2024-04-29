##########################################################
# Softmax in 1D
##########################################################
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0)


# Create class for plotting
def plot_data(data_set, model=None, n=1, color=False):
    X = data_set[:][0]
    Y = data_set[:][1]
    plt.plot(X[Y == 0, 0].numpy(), Y[Y == 0].numpy(), 'bo', label='y = 0')
    plt.plot(X[Y == 1, 0].numpy(), 0 * Y[Y == 1].numpy(), 'ro', label='y = 1')
    plt.plot(X[Y == 2, 0].numpy(), 0 * Y[Y == 2].numpy(), 'go', label='y = 2')
    plt.ylim((-0.1, 3))
    plt.legend()
    if model is not None:
        w = list(model.parameters())[0][0].detach()
        b = list(model.parameters())[1][0].detach()
        y_label = ['yhat=0', 'yhat=1', 'yhat=2']
        y_color = ['b', 'r', 'g']
        Y = []
        for w, b, y_l, y_c in zip(model.state_dict()['weight'], model.state_dict()['bias'], y_label, y_color):
            Y.append((w * X + b).numpy())
            plt.plot(X.numpy(), (w * X + b).numpy(), y_c, label=y_l)
        if color:
            x = X.numpy()
            x = x.reshape(-1)
            top = np.ones(x.shape)
            y0 = Y[0].reshape(-1)
            y1 = Y[1].reshape(-1)
            y2 = Y[2].reshape(-1)
            plt.fill_between(x, y0, where=y1 > y1, interpolate=True, color='blue')
            plt.fill_between(x, y0, where=y1 > y2, interpolate=True, color='blue')
            plt.fill_between(x, y1, where=y1 > y0, interpolate=True, color='red')
            plt.fill_between(x, y1, where=((y1 > y2) * (y1 > y0)), interpolate=True, color='red')
            plt.fill_between(x, y2, where=(y2 > y0) * (y0 > 0), interpolate=True, color='green')
            plt.fill_between(x, y2, where=(y2 > y1), interpolate=True, color='green')
    plt.legend()
    plt.show()


# Create the data class
class Date(Dataset):
    def __init__(self):
        self.x = torch.arange(-2, 2, 0.1).view(-1, 1)
        self.y = torch.zeros(self.x.shape[0])
        self.y[(self.x > -1.0)[:, 0] * (self.x < 1.0)[:, 0]] = 1
        self.y[(self.x >= 1.0)[:, 0]] = 2

        # Method 2
        # self.y = torch.zeros(self.x.size())
        # self.y[(self.x > -1.0) * (self.x < 1.0)] = 1
        # self.y[(self.x >= 1.0)] = 2
        # self.y = self.y[:, 0]

        self.y = self.y.type(torch.LongTensor)
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        # [index] = [index, :], selecting the entire row
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


# Create dataset
data_set = Date()
# plot_data(data_set)
# Build model
# nn.Linear(in_feature, out_feature)
# in_feature: size of each input sample
# out_feature: size of each output sample
model = nn.Linear(1, 3)
# print(model.state_dict())

# Create criterion function, optimizer, and dataloader
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
trainloader = DataLoader(dataset=data_set, batch_size=5)

# Train model
LOSS = []
def train_model(iter):
    for i in range(iter):
        for x, y in trainloader:
            # if iter % 5 == 0:
            #     pass
            #     plot_data(data_set, model)
            yhat = model(x)
            loss = criterion(yhat, y)
            LOSS.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


train_model(400)
plot_data(data_set, model)

# Analyze the result
z = model(data_set.x)
_, yhat = z.max(1)

# Print the accuracy
correct = (yhat == data_set.y).sum().item()
accuracy = correct/len(yhat)
print("The accuracy: ", accuracy)


##############################################################
# Use softmax function to convert the output to a probability
##############################################################
# Create a Softmax object
# softmax(x) = exp(x_i)/sum(exp(x_j))
# dim = -1: the right-most dimension. In this case, it is column
# dim = 1: w.r.t. each row. In other words, sum up the values in each row along the same column
# dim = 2: w.r.t. each column. Sum up the values in each column along the same row
# e.g. x = [[a, b], [c, d]]
# dim = 0: exp(a)/(exp(a) + exp(c))
# dim = 1: exp(a)/(exp(a) + exp(b))
Softmax_fn = nn.Softmax(dim=-1)

# The result is a tensor Probability,
# where each row corresponds to a different sample,
# and each column corresponds to that sample belonging to a particular class.
Probability = Softmax_fn(z)

# We can obtain the probability of the first sample belonging to the first,
# second and third class respectively as follows:
for i in range(3):
    print("probability of class {} is given by  {}".format(i, Probability[0, i]))

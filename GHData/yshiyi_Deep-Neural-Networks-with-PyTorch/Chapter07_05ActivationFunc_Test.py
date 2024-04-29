####################################################################
# Test Sigmoid, Tanh and ReLU Activation Functions
# input dimension: 1
# output dimension: 3, i.e., 0, 1, 2
####################################################################
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pylab as plt
torch.manual_seed(0)

def PlotStuff(X, Y, model=None, leg=False):
    plt.figure()
    plt.plot(X[Y == 0].numpy(), Y[Y == 0].numpy(), 'or',
             label='training points y=0 ')
    plt.plot(X[Y == 1].numpy(), Y[Y == 1].numpy(), 'ob',
             label='training points y=1 ')
    plt.plot(X[Y == 2].numpy(), Y[Y == 2].numpy(), 'og',
             label='training points y=2 ')
    Yhat = model(X)
    _, label = torch.max(Yhat, 1)
    if model is not None:
        plt.plot(X.numpy(), label.numpy(),
                 label='neral network')
    plt.legend()
    plt.show()


# Create dataset
class dataset(Dataset):
    def __init__(self):
        self.x = torch.linspace(-20, 20, 100).view(-1, 1)
        self.y = torch.zeros(self.x.shape[0])
        self.y[(self.x[:, 0] > -10) * (self.x[:, 0] < -5)] = 1
        self.y[(self.x[:, 0] > 5) * (self.x[:, 0] < 10)] = 2
        self.y = self.y.type(torch.LongTensor)
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


# Build the model with sigmoid function
class Net_sig(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net_sig, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        y1 = torch.sigmoid(self.linear1(x))
        yout = self.linear2(y1)
        return yout


# Build the model with Tanh function
class Net_tanh(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net_tanh, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        y1 = torch.tanh(self.linear1(x))
        yout = self.linear2(y1)
        return yout


# Build the model with ReLU function
class Net_relu(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net_relu, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        y1 = torch.relu(self.linear1(x))
        yout = self.linear2(y1)
        return yout


# Define the function for training the model
def train(data_set, model, _criterion, optimizer,
          _train_loader, epochs=500, plot_num=50):
    cost = []
    for epoch in range(epochs):
        total_loss = 0
        for x, y in _train_loader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = _criterion(yhat, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # if epoch % plot_num == 0 or epoch == (epochs - 1):
        #     PlotStuff(data_set.x, data_set.y, model)
        cost.append(total_loss)
    # PlotStuff(data_set.x, data_set.y, model)
    return cost


# Define basic parameters
input_dim = 1
hidden_dim = 2
output_dim = 3
learning_rate = 0.1
# Create dataset
data_set = dataset()
# Create criterion
criterion = nn.CrossEntropyLoss()
# Create data loader
train_loader = DataLoader(dataset=data_set, batch_size=5)

# ##################  Test Sigmoid  ######################
# model_sig = Net_sig(input_dim, hidden_dim, output_dim)
# optimizer_sig = torch.optim.SGD(model_sig.parameters(), learning_rate)
# result_sig = train(data_set, model_sig, criterion,
#                    optimizer_sig, train_loader, 500)

# ##################  Test Tanh  ######################
# torch.manual_seed(0)
# model_tanh = Net_tanh(input_dim, hidden_dim, output_dim)
# optimizer_tanh = torch.optim.SGD(model_tanh.parameters(), learning_rate)
# result_tanh = train(data_set, model_tanh, criterion,
#                     optimizer_tanh, train_loader, 1000)

##################  Test ReLU  ######################
torch.manual_seed(0)
model_relu = Net_relu(input_dim, hidden_dim, output_dim)
optimizer_relu = torch.optim.SGD(model_relu.parameters(), learning_rate)
result_relu = train(data_set, model_relu, criterion,
                    optimizer_relu, train_loader, 1000)


# # Plot results
# plt.figure()
# plt.plot(result_sig, label='Sigmoid')
# plt.plot(result_tanh, label='Tanh')
# plt.plot(result_relu, label='ReLU')
# plt.ylabel('COST')
# plt.xlabel('epochs ')
# plt.legend()
# plt.show()


# Analyze the result of sigmoid
def plot_result(x, y, yhat):
    plt.figure()
    plt.plot(x[y == 0].numpy(),
             y[y == 0].numpy(), 'or', label='training points y=0 ')
    plt.plot(x[y == 1].numpy(),
             y[y == 1].numpy(), 'ob', label='training points y=1 ')
    plt.plot(x[y == 2].numpy(),
             y[y == 2].numpy(), 'og', label='training points y=2 ')
    plt.plot(x.numpy(), yhat.numpy())
    plt.legend()
    plt.show()


# # Sigmoid
# z_sig = model_sig(data_set.x)
# _, y_sig = z_sig.max(1)
# plot_result(data_set.x, data_set.y, y_sig)

# # Tanh
# z_tanh = model_tanh(data_set.x)
# _, y_tanh = z_tanh.max(1)
# plot_result(data_set.x, data_set.y, y_tanh)

# ReLU
z_relu = model_relu(data_set.x)
_, y_relu = z_relu.max(1)
plot_result(data_set.x, data_set.y, y_relu)

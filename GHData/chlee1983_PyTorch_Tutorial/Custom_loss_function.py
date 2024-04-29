import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

x = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [[3], [7], [11], [15]]

X = torch.tensor(x).float()
Y = torch.tensor(y).float()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

X = X.to(device)
Y = Y.to(device)


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]


ds = MyDataset(X, Y)

dl = DataLoader(ds, batch_size=2, shuffle=True)


class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer = nn.Linear(2, 8)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Linear(8, 1)

    def forward(self, x):
        hidden1 = self.input_to_hidden_layer(x)
        hidden2 = self.hidden_layer_activation(hidden1)
        output = self.hidden_to_output_layer(hidden2)
        return output, hidden2


mynet = MyNeuralNet().to(device)


def my_mean_squared_error(_y, y):
    loss = (_y-y)**2
    loss = loss.mean()
    return loss


loss_func = nn.MSELoss()
loss_value = loss_func(mynet(X), Y)
print("Built in loss function ", loss_value)

my_result = my_mean_squared_error(mynet(X), Y)
print("Custom loss function ", my_result)

# fetching the values of intermediate layers
input_to_hidden = mynet.input_to_hidden_layer(X)
hidden_activation = mynet.hidden_layer_activation(input_to_hidden)
print("Hidden activation layer ", hidden_activation)

# the first index output is the hidden layer value post activation
mynet(X)[1]

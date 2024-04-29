##########################################################
# Multiple Outputs Linear Regression:
# Training multiple weighting parameters and biases
##########################################################
import torch
import sys
torch.manual_seed(1)


class LR(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # nn.Linear(in_feature, out_feature)
        # in_feature: size of each input sample
        # out_feature: size of each output sample
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        yhat = self.linear(x)
        return yhat


model = LR(2, 1)
# print(model.state_dict()["linear.weight"].size())
# print(model(torch.tensor([[1.0]])))
# print(model(torch.tensor([[1.0], [2.0], [3.0]])))
# sys.exit(0)

##########################################################
# Training models with multiple outputs
##########################################################
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# sys.exit(0)
class dataset(Dataset):
    def __init__(self):
        self.x = torch.zeros(20, 2)
        self.x[:, 0] = torch.arange(-1, 1, 0.1)
        self.x[:, 1] = torch.arange(-1, 1, 0.1)
        self.w = torch.tensor([[1.0, -1.0], [1.0, 3.0]])
        self.b = torch.tensor([[1.0, -1.0]])
        self.f = torch.mm(self.x, self.w) + self.b
        self.y = self.f + 0.001 * torch.randn((self.x.shape[0], 1))
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


# Create a dataset
data_set = dataset()
# Create a linear regression model
model = LR(2, 2)
# Create an optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# Create loss function
criterion = torch.nn.MSELoss()
# Create a data loader
data_loader = DataLoader(dataset=data_set, batch_size=5)

LOSS = []


def train_model(iter):
    for i in range(iter):
        for x, y in data_loader:
            yhat = model(x)
            loss = criterion(yhat, y)
            LOSS.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


train_model(50)
plt.plot(LOSS)
plt.xlabel("iterations ")
plt.ylabel("Cost/total loss ")
plt.show()


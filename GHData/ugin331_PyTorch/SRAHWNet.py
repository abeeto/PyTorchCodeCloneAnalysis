import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np


# yes, I know these could've been initialized as tensors. no, I don't particularly care
x_data = np.array([-1, 2, 1],dtype='float32')
y_data = np.array([1], dtype='float32')
x_data = Variable(torch.from_numpy(x_data))
y_data = Variable(torch.from_numpy(y_data))


class Net(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        w1 = torch.tensor([[1.0, -2.0, -1.0], [2.0, -1.0, 1.0]])
        with torch.no_grad():
            self.hidden.weight = torch.nn.Parameter(w1)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)
        w2 = torch.tensor([[0.1, 0.0], [0.25, 1.0], [0.5, 2.0], [0.1, 1.0]])
        with torch.no_grad():
            self.hidden2.weight = torch.nn.Parameter(w2)
        self.predict = torch.nn.Linear(n_hidden2, n_output)
        w3 = torch.tensor([[-1.0, -0.5, 0.25, 3.0]])
        with torch.no_grad():
            self.predict.weight = torch.nn.Parameter(w3)

    # swap these for ReLU
    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.hidden2(x))
        x = self.predict(x)
        return x


our_model = Net(n_feature=3, n_hidden=2, n_hidden2=4, n_output=1)

# swap this for CrossEntropyLoss()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(our_model.parameters(), lr=0.05)

# get one forward pass
pred_y = our_model(x_data)
print(pred_y)

# calc loss
loss = criterion(pred_y, y_data)
print(loss)

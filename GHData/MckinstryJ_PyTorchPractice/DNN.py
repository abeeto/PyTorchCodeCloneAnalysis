import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd


class DNN(nn.Module):
    def __init__(self, state_dim=5, action_dim=3):
        """
            Deep Network
        :param state_dim: dimension of input states
        :param action_dim: dimension of action states
        """
        super(DNN, self).__init__()
        self.hidden = nn.Linear(state_dim, 512)
        self.predict = nn.Linear(512, action_dim)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        return self.predict(x)


if __name__ == "__main__":
    states, actions, = 5, 1
    net = DNN(states, actions)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    crit = nn.MSELoss()
    opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # import data
    df = pd.read_csv("./data/AAPL.csv")
    df = df.set_index(["Date"])
    # prep data
    df = df.iloc[:, :-1].pct_change()
    df.iloc[0, :] = 0.0
    x, y = df.iloc[:-1, :], df.iloc[1:, 4]
    # train / test split
    split = .8
    x_train, x_test = x.iloc[:int(len(x) * split), :].values, x.iloc[int(len(x) * split):, :].values
    y_train, y_test = y.iloc[:int(len(y) * split)].values, y.iloc[int(len(y) * split):].values

    # train the network
    epochs = 50
    for ep in range(epochs):
        running_loss = 0.0
        for i, (x, y) in enumerate(zip(x_train, y_train)):
            x, y = torch.tensor(x).to(device), \
                   torch.tensor(y).to(device)
            # zero the parameter gradient buffers
            opt.zero_grad()
            # forward and backward and optimize
            out = net(x.float())
            loss = crit(out, y.float())
            loss.backward()
            opt.step()
            # statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # printing every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (ep + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

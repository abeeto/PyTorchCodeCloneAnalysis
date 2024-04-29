import torch
from torch import nn
import torch.utils.data as Data
import torch.optim as opt

def createDataset(num, num_features):
    w = [3.4, 2.1]
    b = 5.2

    features = torch.randn(num, num_features)
    labels = torch.mm(features, torch.tensor(w).view(-1, 1)) + b
    labels += torch.normal(0, 0.01, size=labels.shape)
    return features, labels


def createDataIter(features, labels, batch_size):
    data = Data.TensorDataset(features, labels)
    data_iter = Data.DataLoader(data, batch_size, shuffle=True)
    return data_iter


class MyNet(nn.Module):
    def __init__(self, num_features):
        super(MyNet, self).__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        y = self.linear(x)
        return y


if __name__ == '__main__':
    num_iters = 3
    num = 1000
    num_features = 2
    batch_size = 10
    lr = 0.03
    net = MyNet(num_features)
    loss = nn.MSELoss()
    optimizer = opt.SGD(net.parameters(), lr=lr)

    features, labels = createDataset(num, num_features)
    data_iters = createDataIter(features, labels, batch_size)

    for epoch in range(1, num_iters + 1):
        for x, labels in data_iters:
            y = net(x)
            l = loss(y, labels)
            l.backward()
            optimizer.step()
            net.zero_grad()
        print(f"epoch = {epoch}, loss = {l.item()}")

    for param in net.parameters():
        print(param)

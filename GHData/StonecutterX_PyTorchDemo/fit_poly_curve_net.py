import numpy as np
import torch
import time

class LineNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.rand(1))
        self.b = torch.nn.Parameter(torch.rand(1))

    def forward(self, x):
        yhat = self.a * x + self.b
        return yhat

class LinearNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        yhat = self.linear(x.unsqueeze(1)).squeeze(1)
        return yhat

def main():
    ## another way
    x = torch.arange(100, dtype=torch.float32) / 100.
    y = 5 * x + 3 + torch.randn(100) * 0.3 # y = 5x+3+n

    net = LineNet()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    for i in range(40000):
        net.zero_grad()
        yhat = net(x)
        loss = criterion(yhat, y)
        loss.backward()

        if i % 1000 == 0:
            print("item = {}, loss = {:.4f}".format(i, loss))

        optimizer.step()

    print(net.a)
    print(net.b)

def main_linear():
    net = LinearNet()
    for p in net.parameters():
        print("p = ", p)

    a = torch.randn([5, 3, 5])
    b = torch.randn([5, 1, 6])
    linear1 = torch.nn.Linear(5, 10)
    linear2 = torch.nn.Linear(6, 10)

    pa = linear1(a)
    pb = linear2(b)
    pc = pa + pb
    d = torch.nn.functional.relu(pc)
    print(a.shape, b.shape)
    print(pa.shape, pb.shape)
    print(pc.shape, d.shape)

def main_test():
    x = torch.randn([500, 10])
    z = torch.zeros([10])

    start = time.time()
    for i in range(500):
        z += x[i]
    print("Took {} seconds.".format(time.time() - start))

    z = torch.zeros([10])
    start = time.time()
    for x_i in torch.unbind(x):
        z += x_i
    print("Took {} seconds.".format(time.time() - start))

    start = time.time()
    z = torch.sum(x, dim=0)
    print("Took {} seconds.".format(time.time() - start))

if __name__ == "__main__":
    # main()
    # main_linear()
    main_test()
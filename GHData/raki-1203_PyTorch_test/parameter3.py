import torch


class PowModel(torch.nn.Module):

    def __init__(self):
        super(PowModel, self).__init__()
        self.powParam = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x, q):
        return self.powParam * x ** q


class LogModel(torch.nn.Module):

    def __init__(self):
        super(LogModel, self).__init__()
        self.logParam = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.logParam * torch.log(x)


class PowLogModel(torch.nn.Module):

    def __init__(self):
        super(PowLogModel, self).__init__()
        self.powModel = PowModel()
        self.logModel = LogModel()

    def forward(self, x, q):
        y = self.powModel(x, q)
        z = self.logModel(y)
        return z


powLogModel = PowLogModel()

z = powLogModel(5.0, 3.0)
print('z', z)


learning_rate = 0.01
for i in range(20):
    powLogModel.zero_grad()
    x = torch.tensor(5.0)
    q = torch.tensor(3.0)
    z = powLogModel(5.0, 3.0)
    target = 10.0
    loss = (z - target) ** 2
    loss.backward()
    with torch.no_grad():
        for name, parameter in powLogModel.named_parameters():
            parameter -= parameter.grad * learning_rate
            print(i, name, parameter.data)
        print(i, 'z', z)

z = powLogModel(5.0, 3.0)
print('z', z)

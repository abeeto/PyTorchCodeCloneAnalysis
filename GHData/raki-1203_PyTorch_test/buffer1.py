import torch


class PowModel(torch.nn.Module):

    def __init__(self):
        super(PowModel, self).__init__()
        self.powParam = torch.nn.Parameter(torch.tensor(1.0))
        self.powBuff = torch.tensor(0.0)

    def forward(self, x, q):
        self.powBuff = self.powBuff * 0.6 + x * 0.4
        return self.powParam * self.powBuff ** q


class LogModel(torch.nn.Module):

    def __init__(self):
        super(LogModel, self).__init__()
        self.logParam = torch.nn.Parameter(torch.tensor(1.0))
        self.logBuff = torch.tensor(0.0)

    def forward(self, x):
        self.logBuff = self.logBuff * 0.6 + x * 0.4
        return self.logParam * torch.log(self.logBuff)


class PowLogModel(torch.nn.Module):

    def __init__(self):
        super(PowLogModel, self).__init__()
        self.powModel = PowModel()
        self.logModel = LogModel()

    def forward(self, x, q):
        y = self.powModel(x, q)
        z = self.logModel(y)
        return z / 2


powLogModel = PowLogModel()

for i in range(10):
    z = powLogModel(5.0, 3.0)
    print(i, 'z', z)
    print(i, 'powBuff', powLogModel.powModel.powBuff)
    print(i, 'logBuff', powLogModel.logModel.logBuff)

print('state', powLogModel.state_dict())

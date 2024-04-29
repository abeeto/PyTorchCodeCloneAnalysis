import torch


class PowModel(torch.nn.Module):

    def __init__(self):
        super(PowModel, self).__init__()

    def forward(self, x, q):
        return x ** q


class LogModel(torch.nn.Module):

    def __init__(self):
        super(LogModel, self).__init__()

    def forward(self, x):
        return torch.log(x)


class PowLogModel(torch.nn.Module):

    def __init__(self):
        super(PowLogModel, self).__init__()
        self.modelList = [PowModel(), LogModel()]

    def forward(self, x, q):
        y = self.modelList[0](x, q)
        z = self.modelList[1](y)
        return z


powLogModel = PowLogModel()

x = torch.tensor(5.0, requires_grad=True)
q = torch.tensor(3.0, requires_grad=True)
z = powLogModel(x, q)
z.backward()

print('x', x)
print('x.grad', x.grad)
print('q', q)
print('q.grad', q.grad)
print('z', z)

print('powLogModel.named_children', list(powLogModel.named_children()))

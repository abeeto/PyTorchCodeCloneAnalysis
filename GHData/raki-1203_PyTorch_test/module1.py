import torch


class PowLogModel(torch.nn.Module):

    def __init__(self):
        super(PowLogModel, self).__init__()

    def forward(self, x, q):
        y = x ** q
        z = torch.log(y)
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

import torch

print(f'torch.__version__ = {torch.__version__}')


class ADG(torch.nn.Module):
    def forward(self, t):
        if t.sum() > 0:
            return t
        else:
            return -t


class ACell(torch.nn.Module):
    def __init__(self):
        super(ACell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h


ac = ACell()
x, h = torch.rand(3, 4), torch.rand(3, 4)
print(f'ac = {ac}')

# Try the traced cell !

tc = torch.jit.script(ac, (x, h))
print(f'tc = {tc}')

print(f'res_a = {ac(x, h)}')
print(f'res_t = {tc(x, h)}')

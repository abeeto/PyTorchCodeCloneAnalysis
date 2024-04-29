import sys
import torch

# Here I train 1 linear layer using higher-level pytorch

# My simple SGD optimizer
class MySGD(torch.optim.Optimizer):
    def __init__(self, params, lr):
        super(MySGD, self).__init__(params, dict(lr=lr))

    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError('closure')
        for group in self.param_groups:
            # print('lr =', group['lr'])
            for p in group['params']:
                p.data.add_(-group['lr'], p.grad.data)

# My simple linear module
class MyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))

    def forward(self, input):
        return input.mm(self.weight.t()) + self.bias

    def extra_repr(self):
        return f'in_features = {self.in_features}, out_features = {self.out_features}'

torch.manual_seed(2014)

device = 'cpu'
# device = 'cuda:0'

d = 10
w_true = torch.tensor([1., 2., 0.5, 3., 4., 1., 2., 1.5, 2., 1.], device=device).reshape(1, d)
b_true = torch.tensor([-0.5], device=device)
print(f'w_true = {w_true}')
print(f'b_true = {b_true}')

# Initialize parameters
w, b = torch.randn(1, d, device=device), torch.randn(1, device=device)
print(f'w = {w}')
print(f'b = {b}')

# net = MyLinear(10, 1)
net = torch.nn.Linear(10, 1)

net.weight.data = w
net.bias.data = b
net.to(device=device)
print('net =', net)

n_iter = 1000
batch_size = 10
lr = 0.1

criterion = torch.nn.MSELoss()

# optimizer = MySGD(net.parameters(), lr=lr)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

# Train
for i in range(n_iter):
    inp = torch.randn(batch_size, d, device=device)
    inp /= inp.norm(dim=1).view(-1, 1)  # Normalize each entry in the batch

    optimizer.zero_grad()

    #    print(inp.shape, w.shape)
    label = inp.mm(w_true.t()) + b_true
    out = net(inp)
    loss = criterion(out, label)
    print(f'{i}: loss = {loss}')

    loss.backward()
    optimizer.step()

print('w = ', w)
print('b = ', b)

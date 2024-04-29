import torch

# Here I train 1 linear layer using low-level pytorch only

torch.manual_seed(2014)

device = 'cpu'
# device = 'cuda:0'

d = 10
w_true = torch.tensor([1., 2., 0.5, 3., 4., 1., 2., 1.5, 2., 1.], device=device).reshape(1, d)
b_true = torch.tensor([-0.5], device=device)
print(f'w_true = {w_true}')
print(f'b_true = {b_true}')

# Initialize parameters
w, b = torch.randn(1, d, requires_grad=True, device=device), torch.randn(1, requires_grad=True, device=device)
print(f'w = {w}')
print(f'b = {b}')
n_iter = 1000
batch_size = 10
lr = 0.1

# Train
for i in range(n_iter):
    inp = torch.randn(batch_size, d, device=device)
    inp /= inp.norm(dim=1).view(-1, 1)  # Normalize each entry in the batch

    #    print(inp.shape, w.shape)
    label = inp.mm(w_true.t()) + b_true
    out = inp.mm(w.t()) + b
    loss = (out - label).pow(2).mean()
    print(f'{i}: loss = {loss}')
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()

print('w = ', w)
print('b = ', b)

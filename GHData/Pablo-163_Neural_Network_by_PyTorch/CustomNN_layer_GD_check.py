import torch

x = torch.tensor([[10., 20.]])
fc = torch.nn.Linear(2, 3)

w = torch.tensor([[11., 12.], [21., 22.], [31., 32]])
fc.weight.data = w
b = torch.tensor([[31., 32., 33.]])
fc.bias.data = b

fc_out = fc(x)
# summing to het scalar function
fc_out_summed = fc_out.sum()

# get grads
fc_out_summed.backward()
weight_grad = fc.weight.grad
bias_grad = fc.bias.grad

# get without FC-layer
w.requires_grad_(True)
b.requires_grad_(True)

our_formula = (x @ w.t() + b).sum()  # SUM{x * w^T + b}

our_formula.backward()

# Checking
print('fc_weight_grad:', weight_grad)
print('our_weight_grad:', w.grad)
print('fc_bias_grad:', bias_grad)
print('out_bias_grad:', b.grad)

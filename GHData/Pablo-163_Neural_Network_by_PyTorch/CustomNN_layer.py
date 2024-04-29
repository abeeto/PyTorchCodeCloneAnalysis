import torch

x = torch.tensor([[10., 20.]])
# torch FC-layer with 2 input and 3 neurons
fc = torch.nn.Linear(2, 3)
# make custom weights and bias
w = torch.tensor([[11., 12.], [21., 22.], [31., 32]])
fc.weight.data = w

b = torch.tensor([[31., 32., 33.]])
fc.bias.data = b

fc_out = fc(x)

# do the same, but with matrix multiplying
fc_out_alternative = x @ w.t() + b

# check
print(fc_out == fc_out_alternative)

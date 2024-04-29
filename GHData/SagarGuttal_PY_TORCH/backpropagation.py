from re import X
import torch

x = torch.tensor(1.0)
y =  torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# forward_pass and compute the loss
y_hat = w * x
loss = (y_hat - y)** 2

print(loss)
# Backword pass
loss.backward()
print(w.grad)

#Next steps
## updated wieghts
## next forward and backword
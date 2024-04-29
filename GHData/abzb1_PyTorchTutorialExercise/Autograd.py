import torch

x = torch.ones(5) # input tensor
y = torch.zeros(3) # expected output
w = torch.randn(5, 3, requires_grad = True)
b = torch.rand(3, requires_grad = True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

loss.backward()
print(w.grad)
print(b.grad)

z = torch.matmul(x, w) + b
print(z.requires_grad) #True

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad) #False

z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)

inp = torch.eye(5, requires_grad = True)
out = (inp + 1).pow(2)
out.backward(torch.ones_like(inp), retain_graph = True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(inp), retain_graph = True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph = True)
print(f"\nCall after zeroing gradients\n{inp.grad}")

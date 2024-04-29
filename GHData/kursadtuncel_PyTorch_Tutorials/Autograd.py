import torch

#x = torch.randn(3, requires_grad=True) # can be false
#print(x)

#y = x+2
#print(y)
#z = y*y*2
#z = z.mean()
#print(z)

#v = torch.tensor([0.1,1.0,0.001], dtype=torch.float32)
#z.backward() # dz/dx
#print(x.grad)

weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
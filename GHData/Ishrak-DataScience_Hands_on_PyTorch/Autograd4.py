import torch
weights=torch.ones(4,requires_grad=True)

for epoch in range(4):
    model_output= (weights*3).sum()
    
    model_output.backward()
    
    print(weights.grad)
    weights.grad.zero_()
import torch

t1 = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32, device='cpu') #'cuda'
print(t1)
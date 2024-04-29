import torch

x = torch.rand(3)
print("Random on CPU:", x)

cuda = torch.device("cuda")
x = torch.rand(3, device=cuda)
print("Random on CUDA:", x)

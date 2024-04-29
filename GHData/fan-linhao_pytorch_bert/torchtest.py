import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
number = torch.cuda.device_count()
print(number)
print(device)
tensor = torch.Tensor(3,4)
tensor.cuda(0)
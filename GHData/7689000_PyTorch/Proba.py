import torch
import torchvision
import numpy as np

# Uni-variate standard Gaussian
torch.manual_seed(1234)
print(torch.randn(4,4))
# random number from uniform distribution
torch.manual_seed(1234)
print(torch.Tensor(4,4).uniform_(0,1))
# Bernoulli by considering uniformly-distributed random values
torch.manual_seed(1234)
print(torch.bernoulli(torch.Tensor(4,4).uniform_(0,1)))
# Uni-variate Gaussian
print(torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1,0,-0.1)))
print(torch.normal(mean=0.5, std=torch.arange(1.,6.)))
print(torch.normal(mean=0.5, std=torch.arange(0.2,0.6)))

a = torch.tensor([10., 10., 13., 10.,
34., 45., 65., 67.,
87., 89., 87., 34.])
print(a)
print(torch.multinomial(a,3))
print(torch.multinomial(a,5, replacement=True))
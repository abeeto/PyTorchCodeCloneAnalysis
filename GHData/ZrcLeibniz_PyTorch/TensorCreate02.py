import torch

# rand/rand_like,randint
# rand会产生0-1之间的数值，不包括1
rand = torch.rand(3, 3)
print(rand)

# rand_like会产生一个与参数维度相同的tensor
like = torch.rand_like(rand)
print(like)

# randint产生一个指定维度和指定数据范围内的int型数据tensor
randint = torch.randint(1, 10, [3, 3])
print(randint)

# randn可以进行正态分布的随机初始化
randn = torch.randn(3, 3)
print(randn)
normal = torch.normal(mean=torch.full([10], 0), std=torch.arange(1, 0, -0.1))
print(normal)

# full
full = torch.full([2, 3], 7)
print(full)

# arange
arange = torch.arange(0, 10)
print(arange)
torch_arange = torch.arange(0, 10, 2)
print(torch_arange)

# linspace/logspace
linspace = torch.linspace(0, 10, 5)
print(linspace)

# ones/zeros/eye
ones = torch.ones(3, 3)
zeros = torch.zeros(3, 3)
eye = torch.eye(3, 3)
print(ones)
print(zeros)
print(eye)

# randperm
randperm = torch.randperm(10)
print(randperm)

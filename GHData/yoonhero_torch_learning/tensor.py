import torch
import numpy as np

# '''
# x = torch.empty(2,2,3)
# print(x)


# x = torch.zeros(2,2)
# print(x)

# x = torch.ones(2, 2, dtype=torch.float16)
# print(x.dtype)


# x = torch.tensor([2.5, 0.1])
# print(x)

# x = torch.rand(2, 2)
# y = torch.rand(2, 2)

# print(x)
# print(y)

# z = x+y
# z = torch.add(x, y)
# print(z)

# x = torch.rand(5, 3)
# print(x)
# print(x[:, 0])

# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(b)

# a.add_(1)
# print(a)
# print(b)

# a = np.ones(5)
# print(a)
# b = torch.from_numpy(a)
# print(b)

# a += 1
# print(a)
# print(b)

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     x = torch.ones(5, device=device)
#     y = torch.ones(5)
#     y = y.to(device)
#     z = x + y

x = torch.ones(5, requires_grad=True)
print(x)
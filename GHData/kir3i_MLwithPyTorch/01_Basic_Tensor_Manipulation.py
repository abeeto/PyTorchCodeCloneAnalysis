import numpy as np
import torch

# 1D Array with NumPy
t = np.array([0, 1, 2, 3, 4, 5, 6])
print(t)

print('Rank of t: ', t.ndim)
print('Shape of t: ', t.shape)

print('t[0], t[1], t[-1] = ', t[0], t[1], t[-1])
print('t[2:5], t[4:-1] = ', t[2:5], t[4:-1])
print('t[:2], t[3:]', t[:2], t[3:])

# 2D Array with NumPy
t = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(t)

print('Rank of t: ', t.ndim)
print('Shape of t: ', t.shape)

# 1D Array with PyTorch
t = torch.FloatTensor([0, 1, 2, 3, 4, 5, 6])
print(t)

print(t.dim())
print(t.shape)
print(t.size()) # same with shape
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])      # can slice like a list
print(t[:2], t[3:])         # can slice like a list

# 2D Array with PyTorch
t = torch.FloatTensor([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                        [10, 11, 12]
                    ])
print(t)

print(t.dim())
print(t.shape)
print(t.size())
print(t[:, 1])  # 0dim - every elements, 1dim - [1]
print(t[:, 1].size())
print(t[:, :-1])    # 0dim - every elements, 1dim - [0] and [1] ( == [:-1])

# Broadcasting
    # PyTorch는 크기가 맞지 않아서 연산할 수 없는 행렬끼리도 (묵시적으로) 크기 변환을 통해 연산이 가능하게 한다.
    # 따라서 debug 과정에 어려움이 있을 수 있으며, 주의를 요한다.
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)  # same dimension, OK

m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3])
print(m1 + m2)  # different dimension, But OK (Broadcast occur)

m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)  # different dimension, But OK (Broadcast occur)

# Multiplication vs. Matrix Multiplication
print()
print('--------------')
print('Mul vs. Matmul')
print('--------------')
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of m1: ', m1.shape)
print('Shape of m2: ', m2.shape)
print(m1.matmul(m2))

m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of m1: ', m1.shape)
print('Shape of m2: ', m2.shape)
print(m1 * m2)      # element wise multiply
print(m1.mul(m2))   # element wise multiply

# Mean
t = torch.FloatTensor([1, 2])
print(t.mean())

t = torch.LongTensor([1, 2])
try:
    print(t.mean())     # Exception occurs
except Exception as exc:
    print(exc)

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

print(t.mean())
print(t.mean(dim=0))
print(t.mean(dim=1))
print(t.mean(dim=-1))

# Sum
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

print(t.sum())
print(t.sum(dim=0))
print(t.sum(dim=1))
print(t.sum(dim=-1))

# Max and Argmax
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

print(t.max())

print(t.max(dim=0))
print('Max: ', t.max(dim=0)[0])
print('Argmax: ', t.max(dim=0)[1])
print(t.max(dim=1))
print(t.max(dim=-1))

# View(Reshape)
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape)

print(ft.view([-1, 3]))     # (4, 3)
print(ft.view([-1, 3]).shape)

print(ft.view([-1, 1, 3]))  # (4, 1, 3)
print(ft.view([-1, 1, 3]).shape)

# Squeeze
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)

print(ft.squeeze())         # (3, )
print(ft.squeeze().shape)

# Unsqueeze
ft = torch.Tensor([0, 1, 2])
print(ft.shape)

print(ft.unsqueeze(0))      # (1, 3)
print(ft.unsqueeze(0).shape)

print(ft.view(1, -1))       # same with unsqueeze
print(ft.view(1, -1).shape)

print(ft.unsqueeze(1))      # (3, 1)
print(ft.unsqueeze(1).shape)

print(ft.unsqueeze(-1))      # (3, 1)
print(ft.unsqueeze(-1).shape)

# Type Casting
lt = torch.LongTensor([1, 2, 3, 4])
print(lt)

print(lt.float())

bt = torch.ByteTensor([True, False, False, True])
print(bt)

print(bt.long())
print(bt.float())

# Concatenate
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])

print(torch.cat([x, y], dim=0)) # (4, 2)
print(torch.cat([x, y], dim=1)) # (2, 4)

# Stacking
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

print(torch.stack([x, y, z]))       # (3, 2)
print(torch.stack([x, y, z], dim=1)) # (2, 3)

print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))   # same with stack([x, y, z])

# Ones and Zeros
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)

print(torch.ones_like(x))
print(torch.zeros_like(x))

# In-place Operation

x = torch.FloatTensor([[1, 2], [3, 4]])

print(x.mul(2))
print(x)
print(x.mul_(2))    # In-place Operation
print(x)
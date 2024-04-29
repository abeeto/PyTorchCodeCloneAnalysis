import torch
import numpy as np

print(torch.zeros([3, 4]))
print(torch.ones([3, 4, 3]))
x = torch.Tensor([[1,2,3,4],
                  [5,6,7,8]])

y = torch.Tensor([[1,2,3,4],
                  [5,6,7,8]])

print(x)
print(x.size())
print(x.shape)
print(x[1])
print(x[0, 0])
print(x[:, 1])
print(x + 10)
print(x ** 2)
print(x + y)
print(x / y)
print(torch.exp(x))

a = torch.ones([5,4])
b = torch.Tensor([[1,2,3,4],
                  [5,6,7,8],
                  [9,10,11,12],
                  [13,14,15,16],
                  [17,18,19,20]])

print(a + b)
print(a % b)
print(a.shape)
print(b[0] + a[2])

a = torch.ones([5,4])
b = torch.Tensor([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16], [17,18,19,20]])
print(a.shape)
print(a + b)
print(b[0] + a[2])
print(a % b)

print(torch.log(a))
print(b > 3)  # constructs mask
print(b[b > 3])  # applies mask b - flattens

y = b  # reference to x
y[0, 0] = 1234
print(y)
print(b)

y = b.clone()  # copy b into y
y[0, 0] = 987
print(y)
print(b)

print(b.dtype)
b = b.double()
print(b.dtype)

x = np.array([[1, 2, 3], [4, 5, 6]])
x = torch.from_numpy(x)
print(x)

print(torch.cuda.is_available())

x = torch.rand([2, 3])
print(x)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

x = x.to(device)  # move tensor to device


x = torch.Tensor([[1, 2], [4, 5]])
print(x)
print(x - 10 * (1/(x+1)))

kernel = torch.tensor([[[[0., 1, 0],
                         [1,  2, 1],
                         [0,  1, 0]],

                        [[1, 2, 1],
                         [0, 3, 3],
                         [0, 1, 10]],

                        [[10, 11, 12],
                         [13, 14, 15],
                         [16, 17, 18]]]])
print(kernel.size())
print(kernel[0, 1, :, :])

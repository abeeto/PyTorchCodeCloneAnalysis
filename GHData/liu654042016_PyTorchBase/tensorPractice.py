from torch import Tensor

#这是一个标量，就是一个值
scalar = Tensor(55)

#正常的数值操作
scalar *=2
#0维
print(scalar.dim())
#可以查看矩阵的维度，如torch.size([2, 3, 4]),且每个维度的大小2， 3， 4
print(scalar.shape)

v = Tensor([1, 2, 3])
print(v.dim())
print(v.size())

matrix = Tensor([[1,2,3], [4, 5, 6]])
print(matrix.dim())
print(matrix.shape)
#矩阵乘法
print(matrix.matmul(matrix.T))
print(matrix.matmul(Tensor([1, 2, 3])))

print(Tensor([1,2]).matmul(matrix))
#对应位置相乘
print(matrix*matrix)
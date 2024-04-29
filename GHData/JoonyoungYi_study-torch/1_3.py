import torch

# x = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
# print(x)

# x_gpu = x.cuda()
# print(x_gpu)
# print(x_gpu.size())

# x_cpu = x_gpu.cpu()
# print(x_cpu)
# print(x_cpu.size())

# x = torch.FloatTensor(10, 12, 3, 3)
# print(x.size())

# x = torch.rand(4, 3)
# print(x)
# out = torch.index_select(x, 1, torch.LongTensor([0, 1]))
# # 변수가 input, dim, index 순이다.
# out = torch.index_select(x, 0, torch.LongTensor([0, 3]))
# print(out)

# x = torch.rand(4, 3)
# print(x)
# print(x[:, 0])
# print(x[0, :])
# print(x[0:2, 0:2])

# x = torch.randn(2, 3)
# mask = torch.ByteTensor([[0, 0, 1], [0, 1, 0]])
# out = torch.masked_select(x, mask)
# print(x)
# print(mask)
# print(out)

# x = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
# y = torch.FloatTensor([[-1, -2, -3], [-4, -5, -6]])
# z1 = torch.cat([x, y], dim=0)
# z2 = torch.cat([x, y], dim=1)
# print(x)
# print(y)
# print(z1)
# print(z2)

# x = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
# x_stack = torch.stack([x, x, x, x], dim=2)
# print(x_stack)

# x = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
# y = torch.FloatTensor([[-1, -2, -3], [-4, -5, -6]])
# z1 = torch.cat([x, y], dim=0)
# x_1, x_2 = torch.chunk(z1, 2, dim=0)
# y_1, y_2, y_3 = torch.chunk(z1, 3, dim=1)
# y__1 = torch.chunk(z1, 1, dim=1)
# print(z1, x_1, x_2, z1, y_1, y_2, y_3)
# print(y__1)

# # torch.split(tensor,split_size,dim=0) -> split into specific size
# x = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
# y = torch.FloatTensor([[-1, -2, -3], [-4, -5, -6]])
# z1 = torch.cat([x, y], dim=0)
# x1, x2 = torch.split(z1, 2, dim=0)
# y1 = torch.split(z1, 2, dim=1)
# print(z1, x1, x2, y1)

# torch.squeeze(input,dim=None) -> reduce dim by 1
# x1 = torch.FloatTensor(10, 1, 3, 1, 4)
# x2 = torch.squeeze(x1)
# print(x1.size(), x2.size())

# x1 = torch.FloatTensor(10, 3, 4)
# x2 = torch.unsqueeze(x1, dim=0)
# print(x1.size(), x2.size())

# import torch.nn.init as init
#
# x1 = init.uniform(torch.FloatTensor(3, 4), a=0, b=9)
# x2 = init.normal(torch.FloatTensor(3, 4), std=0.2)
# x3 = init.constant(torch.FloatTensor(3, 4), 3.1415)
# print(x1, x2, x3)

# x1 = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
# x2 = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
# add = torch.add(x1, x2)
# print(x1, x2, add, x1 + x2, x1 - x2)

# x1 = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
# x2 = torch.add(x1, 10)
# print(x1, x2, x1 + 10, x2 - 10)

# torch.mul() -> size better match
# x1 = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
# x2 = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
# x3 = torch.mul(x1, x2)
# print(x3)

# # torch.mul() -> broadcasting
# x1 = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
# x2 = x1 * 10
# print(x2)

# torch.div() -> size better match

# x1 = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
# x2 = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
# x3 = torch.div(x1, x2)
# print(x3)

# x1 = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
# print(x1 / 5)

# x1 = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
# print(x1 / 5)

# x1 = torch.FloatTensor(3, 4)
# print(torch.pow(x1, 2), x1**2)

# x2 = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
# print(torch.pow(x2, 3))

# x2 = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
# print(torch.exp(x2))

# x1 = torch.FloatTensor(3, 4)
# print(torch.log(x1))

# x1 = torch.FloatTensor(3, 4)
# x2 = torch.FloatTensor(4, 5)
# print(torch.mm(x1, x2))

# x1 = torch.FloatTensor(10, 3, 4)
# x2 = torch.FloatTensor(10, 4, 5)
# print(torch.bmm(x1, x2).size())

# # torch.dot(tensor1,tensor2) -> dot product of two tensor
# x1 = torch.FloatTensor([1, 2, 3])
# x2 = torch.FloatTensor([4, 5, 6])
# print(torch.dot(x1, x2))

# x1 = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
# x2 = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
# print(torch.dot(x1, x2))

# x1 = torch.FloatTensor(3, 4)
# print(x1, x1.t())

# x1 = torch.FloatTensor(10, 3, 4)
# print(x1.size(), torch.transpose(x1, 1, 2).size(), x1.transpose(0, 2).size())

# torch.eig(a,eigenvectors=False) -> eigen_value, eigen_vector
x1 = torch.FloatTensor([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
# print(x1, torch.eig(x1, True))
print(torch.eig(x1, True))

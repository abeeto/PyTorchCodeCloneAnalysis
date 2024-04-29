import torch
import numpy as np

x = torch.Tensor(3, 4)  # 三行四列的一个tensor
print(x)
y = torch.Tensor([1, 2, 3, 4])  # 指定列表生成的tensor
print(y)
print('-' * 100)

a = torch.FloatTensor(2, 3)
b = torch.FloatTensor([2, 3, 4, 5])
print(a)
print(b)
print('-' * 100)

a = torch.IntTensor(2, 3)
b = torch.IntTensor([2, 3, 4, 5])
# 一维张量（矩阵的列向量）
c = torch.IntTensor([[2, 3, 4, 5], [2, 3, 4, 5]])
# 二维张量（矩阵）
print(a)
print(b)
print(c)
print('-' * 100)

aa = torch.rand(2, 3)  # 用于生成数据类型为浮点型且维度指定的随机Tensor
print(aa)
bb = torch.randn(2, 3)
# 用于生成数据类型为浮点型且维度指定的随机Tensor,
# 随机生成的浮点数的取值满足均值为0,方差为1的正态分布
print(bb)
print('-' * 100)

c = torch.arange(1, 20, 1)
# 用于生成数据类型为浮点型且自定义起始范围和结束范围的Tensor
print(c)
print('-' * 100)

a = torch.ones(2, 3)
print(a)
a = torch.zeros(3, 3)
# 用于生成数据类型为浮点型且维度指定的Tensor，不过这个浮点型
# 的Tensor中的元素值全部为0
print(a)

a = torch.randn(2, 3)
print(a)
b = torch.abs(a)
# 将参数传递到torch.abs后返回输入参数的绝对值作为输出，输入参数
# 必须是 Tensor数据类型的变量。
print(b)
print('-' * 100)
a = torch.randn(2, 3)
b = torch.randn(2, 3)
print(a)
print(b)
c = torch.add(a, b)
# 将参数传递到 torch.add 后返回输入参数的求和结果作为输出，输入
# 参数既可以全部是Tensor数据类型的变量，也可以一个是 Tensor 数据类型的变量，
# 另一个是标量。
print(c)
d = torch.randn(2, 3)
print(d)
e = torch.add(d, 10)
print(e)
print('-' * 100)

a = torch.randn(2, 3)
print(a)
b = torch.clamp(a, -0.1, 0.1)
# 对输入参数按照自定义的范围进行裁剪,最后将参数裁剪的结果作
# 为输出。所以输入参数一共有三个，分别是需要进行裁剪的 Tensor 数据类型的变量、裁剪
# 的上边界和裁剪的下边界， 具体的裁剪过程是：使用变量中的每个元素分别和裁剪的上边
# 界及裁剪的下边界的值进行比较，如果元素的值小于裁剪的下边界的值，该元素就被重写
# 成裁剪的下边界的值；同理，如果元素的值大于裁剪的上边界的值 该元素就被重写成裁
# 剪的上边界的值。
print(b)

a = torch.randn(2, 3)
print(a)
b = torch.randn(2, 3)
print(b)
c = torch.div(a, b)
# 将参数传递到 torch.div后返回输入参数的求商结果作为输出，同样，
# 参与运算的参数可以全部是Tensor数据类型的变量，也可以是Tensor数据类型的变量和
# 标量的组合。
print(c)
d = torch.randn(2, 3)
print(d)
e = torch.div(d, 10)
print(e)
print('-' * 100)

a = torch.randn(2, 3)
print(a)
b = torch.randn(2, 3)
print(b)
c = torch.mul(a, b)
# 将参数传递到torch.mul后返回输入参数求积的结果作为输出， 参与
# 运算的参数可以全部是 Tensor 数据类型的变量,也可以是Tensor数据类型的变量和标量
# 的组合。
print(c)
d = torch.randn(2, 3)
print(d)
e = torch.mul(d, 10)
print(e)
print('-' * 100)

d = torch.randn(2, 3)
print(d)
e = torch.pow(d, 2)
# 将参数传递到torch.pow 后返回输入参数的求幕结果作为输出参与
# 运算的参数可以全部是Tensor数据类型的变量，也可以是Tensor数据类型的变量和标盘
# 的组合
print(e)
print('-' * 100)

a = torch.randn(2, 3)
print(a)
b = torch.randn(3, 2)
print(b)
c = torch.mm(a, b)
# 将参数传递到torch.mm后返回输入参数的求积结果作为输出，不过
# 这个求积的方式和之前的 torch.mul的运算方式不太一样，torch.mm 运用矩阵
# 之间的乘法规则进行计算 ，所以被传入的参数会被当作矩阵进行处理，参数的维度
# 自然也要满足矩阵乘法的前提条件 ，即前一个矩阵的行数必须和后
# 个矩阵的列数相等，否则不能进行计算。
print(c)
print('-' * 100)

a = torch.randn(2, 3)
print(a)
b = torch.randn(3)
print(b)
c = torch.mv(a, b)
# 将参数传递到Torch.mv后返回输入参数的求积结果作为输出,Torch.mv
# 运用矩阵与向量之间的乘法规则进行计算,被传入的参数中的第1个参数代表矩阵，第
# 2个参数代表向量，顺序不能颠倒。
# 对矩阵a和向量b进行相乘。 如果a是一个n×m张量，b是一个m元 1维张量，
# 将会输出一个n元1维张量。
# 标量是零维张量，向量是一维度张量（在矩阵中为一个列向量）,矩阵是二维度张量，还有更高维的张量
print(c)
print('-' * 100)

b = torch.randn(3)
# 一个向量（矩阵的列向量，输出格式有一个中括号）
d = torch.randn(1, 3)
# 一个矩阵（输出格式有两个中括号）
print(b)
print(d)
print('-' * 100)

a = torch.randn(3, 6)
print(a)
print(a.size()[0])
b = a.view(a.size()[0], -1)
print(b)
print("-" * 100)

a = torch.arange(0, 6)
a = a.view(2, 3)
print(a)
b = a.view(-1, 3)
print(b)
b = b.unsqueeze(1)
# 注意形状，在第1维（下标从0开始）上增加“1”
print(b)
b = b.unsqueeze(-2)
# -2表示倒数第2个维度
print(b)
# 几个中括号则为几维
c = b.view(1, 1, 1, 2, 3)
c.squeeze(0)
# 压缩第0维的“1”
print(c)
c.squeeze()
# 压缩所有维度的“1”
print(c)
print("-" * 100)

# 索引操作
a = torch.randn(3, 4)
print(a)
print(a[0])
# 输出第0行
print(a[:, 0])
# 输出第0列
print(a[0][2])
# 输出第0行第2个元素
print(a[0][-1])
# 输出第0行最后一个元素
print(a[:2])
# 输出前两行
print('-' * 100)

a = np.ones([2, 3])
print(a)
b = torch.ones([2, 3])
print(b)
c = torch.from_numpy(a)
print(c)
# 对于X[:,0];
# 是取二维数组中第一维的所有数据
# 对于X[:,1]
# 是取二维数组中第二维的所有数据
# 对于X[:,m:n]
# 是取二维数组中第m维到第n-1维的所有数据
# 对于X[:,:,0]
# 是取三维矩阵中第一维的所有数据
# 对于X[:,:,1]
# 是取三维矩阵中第二维的所有数据
# 对于X[:,:,m:n]
# 是取三维矩阵中第m维到第n-1维的所有数据
print('-' * 100)

a = torch.rand(3, 1)
print(a)
a = torch.rand(1, 3)
print(a)
print('-' * 100)

np_data = np.array([1, 1, 1])
torch_data = torch.from_numpy(np_data)  # numpy转换为torch
tensor2array = torch_data.numpy()  # torch转换为numpy
print('-' * 100)

aaa = np.array([1, 1, 1])
print(aaa.shape[0])
# 行

# 列
a = np.array([1, 2, 3, 4])
print(a)
print(a.shape)
# a是一个向量,默认是列向量

b = np.array([[1, 2, 3, 4]])
print(b)
print(b.shape)
# b是一个矩阵
print('-' * 100)

b = torch.randn(3)
# 一个向量（矩阵的列向量，输出格式有一个中括号）
d = torch.randn(1, 3)
# 一个矩阵（输出格式有两个中括号）
print(b)
print('-' * 100)

a = torch.arange(1, 17)

print(a)
print(a.size(0))

b = a.view(a.size(0), -1)
print(b)
c = a.view(a.size(0), 1)
print(c)

d = a.view(-1, 16)
print(d)
e = a.view(-1, a.size(0))
print(e)
print('-' * 100)

a = torch.randn(3, 1)
print(a)
b = torch.randperm(3)
a = a[b]
print(a)
print('-' * 100)

b = torch.ones(8) * 3
print(b)
print(b.size(0))
print(b.long())
print('-' * 100)

a = torch.randn(2, 4, 4)
print(a)
b = torch.mean(a)
print(b)
print('-' * 100)

a = torch.randn(2, 3)
print(a)
x = torch.randn(2, 3, 4, 4)
print(x)
a = a.view(2, 3, 1, 1)
print(a)
a = a.repeat(1, 1, 4, 4)
print(a)
x = torch.cat([x, a], dim=1)
print(x)
print('-' * 100)
import torch

# 生成一个3行3列的矩阵
t0 = torch.arange(1, 10).reshape(3, 3)
# 表示索引第一行第二个元素
print(t0[0, 1])
# 表示索引第一行，每隔两个元素取一个
print(t0[0, ::2])
# 索引位置传一个list，代表一次性索引多个值，即在第一行，索引第1和第三个元素
print(t0[0, [0, 2]])
# 每隔2行取一行，且每一行中每隔2个元素取一个
print(t0[::2, ::2])

# 生成3X3的矩阵，3维
t1 = torch.arange(1, 28).reshape(3, 3, 3)
print(t1)
# 索引第二个矩阵中，第二行，第二个元素
print(t1[1, 1, 1])
# 同二维，索引第二个举证，行列都每隔2个元素取一个
print(t1[1, ::2, ::2])
print(t1[::2, ::2, ::2])

# ------------函数索引
t2 = torch.arange(1, 28).reshape(9, 3)
print(t2)
indices = torch.tensor([1, 2])
# 将t2，从第一个维度进行索引[1,2]，即返回2，3
# 高维时用起来比较方便，且意思清晰
# lesson 2 第26分钟
print(torch.index_select(t2, 0, indices))

t3 = torch.arange(6).reshape(2, 3)
print(t3)
# 构建一个数据相同，但形状不同的“视图”
t4 = t3.view(3, 2)
print(t4)

# view还可以修改维度
# 创建了一个3维张量
t5 = t4.view(1, 2, 3)
print(t5)

# ------------张量的分片函数,split返回的也是一个视图，因此是原本对象的引用
t6 = torch.arange(12).reshape(4, 3)
# 在0维度上，按行，进行四等分,返回的也是一个“视图”。不是新生成的对象。
print(torch.chunk(t6, 4, dim=0))
# 第二个参数输入一个整数时，是按整数均分
print(torch.split(t6, 2, 0))
# 按索引进行切分
# 按1:3的比例进行切分
print(torch.split(t6, [1, 3], dim=0))
# 分成四份，每份都是1个
print(torch.split(t6, [1, 1, 1, 1], dim=0))

# -------------张量的合并

a = torch.zeros(2, 3)
b = torch.ones(2, 3)
c = torch.zeros(3, 3)
# dim 默认取值为0
print(torch.cat([a, b]))
# 按dim=1的维度进行拼接
print(torch.cat([a, b], 1))
# c铁定不能和ab拼接，形状不匹配


# --------------堆叠函数
# 将A,B对堆到一个三维张量中去
print(torch.stack([a, b]))

# --------------维度变换
# squeeze可以用来剔除不重要的维度
# unsqueeze 可以用来升维

# 创建了一个全0 4维张量，
# 一个包含一个三维的四维张量，
# 三维张量只包含一个三行一列的二维张量
t7 = torch.zeros(1, 1, 3, 1)

# 在第1个维度升维，也是第0个索引位上升维
print(torch.unsqueeze(t7, dim=0))

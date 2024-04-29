import torch

# import numpy as np

# 矩阵的形变与特殊矩阵的构造方法
'''
 torch.t t转置
 .eye 创建包含N隔分量的单位矩阵
 .diag 以t1中各元素，创建对角矩阵
 .triu 取矩阵t中的上三角矩阵
 .tril 取矩阵t中的下三角矩阵
'''

# 创建单位矩阵，相当于1，单位矩阵左乘或者右乘一个矩阵A，结果都为A
torch.eye(3)

# 对角线矩阵
t1 = torch.arange(4)
torch.diag(t1)

# 1 对角线上元素统一往右上方移动一位, -1左下方偏移一位
torch.diag(t1, 1)

tt = torch.arange(9).reshape(3, 3)
# 取上三角矩阵
torch.triu(tt)
# 上三角矩阵向左下偏移一位
torch.triu(tt, -1)
# 上三角矩阵向右上偏移一位
torch.triu(tt, 1)

# 取下三角矩阵
torch.tril(tt)

# 矩阵的基本运算
'''
 torch.dot 计算张量内积
 .mm 矩阵乘法
 .mv 矩阵乘向量
 .bmm 批量矩阵乘法
 .addmm(input,mat1,mat2,beta=1,alpha=1) 矩阵相乘后相加
    输出结果 = beta * input + alpha * (mat1 * mat2)
 .addbmm 批量矩阵相乘后相加
'''

# dot/vdot 点积计算
'''
注意，在pytorch中，dot和vdot只能作用于一维张量，且对于数值型对象，
二者计算结果并没有区别，两种函数只在进行复数运算时，会有区别。

点积，即两个向量对应位置的元素相乘后相加
'''
t2 = torch.arange(1, 4)
torch.dot(t2, t2)
torch.vdot(t2, t2)

t3 = torch.arange(1, 7).reshape(2, 3)
t4 = torch.arange(1, 10).reshape(2, 3)
# *符号，对应位置相乘
t3 * t4

# 矩阵乘法,回顾：左乘矩阵的列数必须等于右乘矩阵的行数
# 乘法过程，A*B：A每行乘B每列后相加
torch.mm(t3, t4)

# 矩阵和向量相乘
met = torch.arange(1, 7).reshape(2, 3)
vec = torch.arange(1, 4)
# 实际执行过程中，需要矩阵的列数和向量的元素个数相同
# 过程：可以把矩阵看成N个向量构成的矩阵。然后矩阵中每个行向量和被乘向量做点积
torch.mv(met, vec)
# 同时等价于：把被乘向量转换成列向量然后左乘矩阵
torch.mm(met, vec.reshape(3, 1))
torch.mm(met, vec.reshape(3, 1)).flatten()
'''
 mv的理解
 其本质上提供了一种二维张量于一维张量相乘的方法，在线性代数运算
 过程中，有很多矩阵乘向量的场景，典型的如线性回归的求解过程，通常
 情况下我们需要将向量转化为列向量，然后进行计算，但pytorch中单独
 设置了一个矩阵和向量乘的方法，从而简化了行/列向量的理解过程和将
 向量转化为列向量的转化过程。
'''

# bmm 批量矩阵的相乘，比如三维矩阵
t5 = torch.arange(1, 13).reshape(3, 2, 2)
t6 = torch.arange(1, 19).reshape(3, 2, 3)

torch.bmm(t5, t6)

# 矩阵的代数运算
'''
 torch.trace 矩阵的迹
 .matrix_rank 矩阵的秩
 .det 计算矩阵的行列式
 .inverse 矩阵求逆
 .lstsq 最小二乘法 
'''

# 矩阵的迹
'''
迹运算就是矩阵对角线元素之和，在pytorch中，可以使用trace函数完成
'''
a = torch.tensor([[1, 2], [3, 4]]).float()
print(torch.trace(a))

# 矩阵不一定需要方阵,如下即1+6
b = torch.arange(1, 7).reshape(2, 3)
print(torch.trace(b))

# 矩阵的秩（rank）
'''
指的矩阵中行或者列的极大线性无关数(视频35分钟左右)，且矩阵中行、列极大无关数总是相同的，
任何矩阵的秩都是唯一值，满秩指的是方阵（行数和列数相同的矩阵）中行数、列数
和秩相同，满秩矩阵有线性唯一解等重要特性，而其他矩阵也能通过求解秩来降维，
同时，秩也是奇异值分解等运算中涉及到的重要概念。

'''
a1 = torch.arange(1, 5).reshape(2, 2).float()
torch.matrix_rank(a1)
# 对于矩阵b1，第一列和第二列明显线性相关，最大线性无关组只有1组，因此
# 矩阵的秩计算结果为1
b1 = torch.tensor([[1, 2], [2, 4]]).float()
torch.matrix_rank(b1)

# 矩阵的行列式，必须是方阵
# 回顾：即代数余子式的求和过程
# 用来区分取值为0和不为0的行列式
# 不满秩的行列式结果为0，如果是0的话就无法计算矩阵的逆矩阵
# 备注学习点：对角线法计算行列式（视频43分）

a2 = torch.tensor([[1, 2], [4, 5]]).float()
torch.det(a2)

# 线性方程组的矩阵表达形式
'''
回顾，逆矩阵定义：如果存在两个矩阵A、B，并在矩阵乘法运算下，A*B=E（单位阵）
则我们称A、B互为逆矩阵。
'''
a3 = torch.tensor([[1.0, 1], [3, 1]])
b3 = torch.tensor([2.0, 4])
torch.inverse(a3)
# 逆矩阵和本身相乘返回单位阵E ("1")
torch.mm(torch.inverse(a3), a3)

# 案例 求解 一元二次方程线性方程组，视频（55分）
'''
A^-1 * A * x = A^-1 * B   PS： A^-1 * A 消掉，得单位阵E
E * x = A^-1 * B
x = A^-1 * B
即 👇
'''
c = torch.mv(torch.inverse(a3), b3)
# c的元素，即是对方程系数的解，即 y=ax+b，中的a和b。


# 矩阵的分解 （SVD分解/降维）
'''
任何矩阵A 可以看成   A = A * A^-1 * A，也可以看成某种意义上的分解。
大多数形况下，矩阵分解：A = V * U * D
'''
# 特征分解
'''
特征分解中，矩阵分解形式为：
    A = Q * λ * Q^-1
其中Q和Q^-1互为逆矩阵，并且Q的列就是A的特征值对应的特征向量，而λ为矩阵
A的特征值按照降序排列组成的对角矩阵。
torch.eig：特征分解
'''
'''
特征值：矩阵的一列所包含的信息量。见视频1小时处讲解的eig函数部分
'''
a4 = torch.arange(1, 10).reshape(3, 3).float()
# 此处eigenvectors=True会返回完整的 Q 和 λ。
torch.eig(a4, eigenvectors=True)
# 返回结果中，eigenvalues是特征向量，即A矩阵分解后的λ矩阵的对角线元素值。
# eigenvectors是A矩阵分解后的Q矩阵。

b4 = torch.tensor([1, 2, 2, 4]).reshape(2, 2).float()
torch.matrix_rank(b4)
torch.eig(b4)

c1 = torch.tensor([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
# 特征值结果 第二第三个几乎接近于0，因此能够表示，该矩阵其实用一列的线性关系就能把其他列表示出来。
# 这就是为了识别出高有效的列，后面用于降维->奇异值分解（SVD）
torch.eig(c1)

# 奇异值分解SVD
torch.svd(c1)
# 验证奇异值分解，通过对角线矩阵验证
CU, CS, CV = torch.diag(c1)
# 奇异值还原原矩阵
torch.mm(torch.mm(CU, torch.diag(CS)), CV.t())
# 降维，根据SVD输出结果，进行降维
# 操作：索引出所有的行，第0列。然后变成三行一列向量
U1 = CU[:, 0].reshape(3, 1)
# 同理
V1 = CV[:, 0].reshape(1, 3)
# 证明，同样能还原原矩阵
C1 = CS[0]
torch.mm((U1 * C1), V1)

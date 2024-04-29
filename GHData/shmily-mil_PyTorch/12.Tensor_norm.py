'''

norm:范数

范数：常用来度量某个向量空间（或矩阵）中的每个向量的长度或大小

0范数:指当前的向量中非零元素的个数和
1范数:差值绝对值的和
2范数:差值的平方和，在开根号，就是欧氏距离
p范数:
核范数:低秩问题的求解

'''

import torch

a = torch.rand(1,1)
b = torch.rand(1,1)

print(a,b)

print(torch.dist(a,b,p=1))
print(torch.dist(a,b,p=2))
print(torch.dist(a,b,p=3))

print(torch.norm(a))
print(torch.norm(a,p="fro")) # 计算核范数
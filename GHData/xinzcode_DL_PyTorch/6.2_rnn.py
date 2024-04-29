import torch

# 构造矩阵X、W_xh、H和W_hh，它们的形状分别为(3, 1)、(1, 4)、(3, 4)和(4, 4)。
X, W_xh = torch.randn(3, 1), torch.randn(1, 4)
H, W_hh = torch.randn(3, 4), torch.randn(4, 4)
# 将X与W_xh、H与W_hh分别相乘，再把两个乘法运算的结果相加，得到形状为(3, 4)的矩阵
print(torch.matmul(X, W_xh) + torch.matmul(H, W_hh))
# 将矩阵X和H按列（维度1）连结，连结后的矩阵形状为(3, 5)。可见，连结后矩阵在维度1的长度为矩阵X和H在维度1的长度之和（1+41+4）。
# 然后，将矩阵W_xh和W_hh按行（维度0）连结，连结后的矩阵形状为(5, 4)。
# 最后将两个连结后的矩阵相乘，得到与上面代码输出相同的形状为(3, 4)的矩阵
print(torch.matmul(torch.cat((X, H), dim=1), torch.cat((W_xh, W_hh), dim=0)))

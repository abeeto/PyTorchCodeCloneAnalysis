import torch

a = torch.rand(2,2)
print(a)

# 通过对某个维度求值，可以达到降维的效果

# torch.mean()
print(torch.mean(a,dim=0))

# torch.sum()
print(torch.sum(a,dim=0))

# torch.prod() # 计算所有元素的积
print(torch.prod(a,dim=0))

# torch.max()
# torch.min()
# torch.argmax() # 返回最大值排序的索引值
print(torch.argmax(a))

# torch.argmin() # 返回最小值排序的索引值
print(torch.argmin(a))

# torch.std() # 返回标准差
print(torch.std(a))

# torch.var() # 返回方差
print(torch.var(a))

# torch.median() # 返回中间值
print(torch.median(a))

# torch.mode() # 返回众数值
print(torch.mode(a))

# torch.histc() # 计算input的直方图
b = torch.rand(2,2) * 10
print(b)
print(torch.histc(b,6,0,0))

# torch.bincount() # 返回每一个值的帧数,只支持一维tensor
c = torch.randint(0,10,[10])
print(c)
print(torch.bincount(c))
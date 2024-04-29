import torch

#不初始化
print(torch.empty(5,3))

#随机初始化5x3矩阵
print(torch.rand(5,3))

#全为0，long类型
print(torch.zeros(5,3,dtype=torch.long))

x = torch.tensor([5,2])
print(x)

#从一个tensor，构造另一个tensor
x = x.new_ones(5,3,dtype=torch.double)
print(x)

x = torch.rand_like(x,dtype=torch.double)   #float是CPU用的，GPU用double
print(x) #维度一致
print(x.size()) #是一个元组

y = torch.rand(5,3,dtype=torch.double)

print(x+y)
result = torch.zeros(5,3,dtype=torch.double) #GPU版本必须有dtype
torch.add(x,y,out=result)
print(result)

y.add_(x) #add x to y
print(y[:,1]) #第一列（从第0列开始数）
print(y[1,:]) #第一行

x = torch.rand(4,4)
y = x.view(1,16) #重构大小
z = x.view(-1,8)
print(x,y,z)

x = torch.rand(2) #1x2
print(torch.rand(1).item()) #只有1x1有

import torch

x = torch.ones(2,2,requires_grad=True) # 生成2*2的矩阵  默认初始值为1
print(x)

y = x + 2
z = y *y + 3
out = z.mean()
print(y)
print(z)
print(out)

# 计算梯度
out.backward()

print(x.grad)

with torch.no_grad():  # 忽略梯度
    k = x + 1
    print(k)
    
    
k = x + 1
print(k)
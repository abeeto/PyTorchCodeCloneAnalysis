import torch

# 定义向量
vectro = torch.tensor([1,2,3,4])
print("Vector:\t\t",vectro)
print('Vector: Shape:\t',vectro.shape)

# 定义矩阵
matrix = torch.tensor([[1,2],[3,4]])
print('Matrix:\n',matrix)
print('Matrix Shape:\n',matrix.shape)

# 定义张量
tensor = torch.tensor([ [ [1,2],[3,4] ], [ [5,6],[7,8] ]  ])
print('Tensor:\n',tensor)
print('Tensor Shape:\n',tensor.shape)

# Autograd  完成所有的梯度下降和反向传递
# 在autograd下 反向传递(backprop)代码自动定义

    # .requires_grad
        # 在tensor上设定.requires_grad=true后,autograd会自动跟踪所有与改tensor有关的所有运算
    # .backward()
        # 所有运算完成后，执行.backward(),,autograd会自动计算梯度并执行反向传递
    # .grad
        # 用来访问梯度
    # with torch.no_grad()
        # 自动忽略梯度

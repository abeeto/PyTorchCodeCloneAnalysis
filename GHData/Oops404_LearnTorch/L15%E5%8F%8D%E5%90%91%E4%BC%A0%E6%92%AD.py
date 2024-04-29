# -*- coding: UTF-8-*-
import torch
import torch.nn as nn
from torch.nn import functional as f

"""
@Project: LearnTorch
@Author: Oops404
@Email: cheneyjin@outlook.com
@Time: 2022/1/22 19:30
"""
"""
求导路径
    w -Σ-> z -sigmoid()-> sigma -BCEloss()-> loss 
         <----------反过来求导，即得w--------

    ∂Loss / ∂w，其中
    Loss = -Σ(i = 1~m)(y_i * ln(σ_i) + (1 - y_i) * ln(1 - σ_i))
这是单层，如果是双层、n层呢？多层传递后公式多层嵌套再偏导，极其复杂/(ㄒoㄒ)/。
    
反向传播算法BP：
链式求导救我！(ง •_•)ง
假设有函数 u=h(z), z=f(w), 且两函数在各自自变量的定义域上都可导，则有：
    ∂u/∂w = ∂u/∂z * ∂z/∂w
这里用↑表示神经网络表达式中的层号， ↑层↓序。
若网络有两层：
    ∂Loss / ∂w↑(1→2) = ∂L(σ) / ∂σ * ∂σ(z) / ∂z * ∂z(w) / ∂w
    
    ∂L(σ) / ∂σ = ∂(-Σ(i=1~m) (y_i * ln(σ_i) + (1 - y_i) * ln(1 - σ_i))) / ∂σ
               = Σ(i=1~m) (∂(-(y_i * ln(σ_i) + (1 - y_i) * ln(1 - σ_i))) / ∂σ)
前进提要，求导不影响加和，因此先加和还是先求导无所谓，先不看它。
    =-(y * 1 / σ + (1 - y) * 1 / (1 - σ) * (-1))
    =-(y / σ + (y - 1) / (1 - σ))
    =-(y(1 - σ) + (y - 1)σ) / (σ(1 - σ))
    =-(y - yσ + yσ - σ) / σ(1 - σ)
    =(σ - y) / σ(1 - σ)
    
    ∂σ(z) / ∂z = ∂(sigmoid(z)) / ∂z
               = ∂(1 / (1 + e^(-z))) / ∂z
               = ∂((1 + e^(-z))^(-1)) / ∂z
               = -1 * (1 + e^(-z))^(-2) * e^(-z) * (-1)
               = e^(-z) / (1 + e^(-z))^2
             
               = (1 + e^(-z) - 1) / (1 + e^(-z))^2
               = ((1 + e^(-z)) / (1 + e^(-z))^2) - (1 / (1 + e^(-z))^2)
               = (1 / (1 + e^(-z))) - (1 / (1 + e^(-z))^2)
               = (1 / (1 + e^(-z))) * (1 - (1 / (1 + e^(-z))))
               = σ(1 - σ)
此时的σ还是第二层的σ，因此接着：
    ∂z(w) / ∂w = ∂σ↑(1)w / ∂w = σ↑(1)

综上所述：
    ∂Loss / ∂w↑(1→2) = ∂L(σ) / ∂σ * ∂σ(z) / ∂z * ∂z(w) / ∂w
                     = (σ↑(2) - y) / (σ^2 * (1 - σ↑(2))) * σ↑(2) * (1 - σ↑(2)) * σ↑(1)
                     = σ↑(1) * (σ↑(2) - y)
    ∂Loss / ∂w↑(0→1) = ∂L(σ) / ∂σ↑(2) * ∂σ(z) / ∂z↑(2) * ∂z(σ) / ∂σ↑(1) * ∂σ(z) / ∂z↑(1) * ∂z(w) / ∂w(0→1)
                     = (σ↑(2) - y) * ∂z(σ) / ∂σ↑(1) * ∂σ(z) / ∂z↑(1) * ∂z(w) / ∂w(0→1)
                     = (σ↑(2) - y) * w↑(1→2) * (σ↑(1) * (1 - σ↑(1))) * X
"""
# ----------------------------------------------------------------------------------
# x = torch.tensor(1., requires_grad=True)
# y = torch.tensor(2., requires_grad=True)
# z = x ** 2
# sigma = torch.sigmoid(z)
# loss = -(y * torch.log(sigma) + (1 - y) * torch.log(1 - sigma))
# grad = torch.autograd.grad(loss, x)
# print(grad)
# ----------------------------------------------------------------------------------

# 生成500个随机样本，每个样本20个特征
X = torch.rand((500, 20))
# 三分类问题，我们吧0、1、2代指分类结果。 结果为500行1列。
y = torch.randint(low=0, high=3, size=(500,), dtype=torch.float32)

input_ = X.shape[1]
output_ = len(y.unique())

torch.manual_seed(996)


class Model(nn.Module):
    """
    参见L12
    """

    def __init__(self, in_features=10, out_features=2):
        super().__init__()
        self.linear1 = nn.Linear(in_features, 13, bias=False)
        self.linear2 = nn.Linear(13, 8, bias=False)
        self.output = nn.Linear(8, out_features, bias=True)

    def forward(self, x):
        sigma1 = torch.relu(self.linear1(x))
        sigma2 = torch.sigmoid(self.linear2(sigma1))
        zhat = self.output(sigma2)
        return zhat


net = Model(in_features=input_, out_features=output_)
zhat = net.forward(X)
criterion = nn.CrossEntropyLoss()
loss = criterion(zhat, y.long())
# 反向传播
loss.backward()  # retain_graph = True 重复计算
print(net.linear1.weight.grad)
print(net.linear2.weight.grad)

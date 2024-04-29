import torch
from torch import nn


# 定义一个不含模型参数的自定义层
# 定义了一个将输入减掉均值后输出的层，并将层的计算定义在了forward函数里。这个层里不含模型参数。
class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()


layer = CenteredLayer()
print(layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)))

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
# 下面打印自定义层各个输出的均值。因为均值是浮点数，所以它的值是一个很接近0的数。
y = net(torch.rand(4, 8))  # 同时初始化两个层
# print(y[1])
print(y.mean().item())


# 自定义含模型参数的自定义层，应将参数定义成Parameter，还可以使用ParameterList和ParameterDict
class MyListDense(nn.Module):
    def __init__(self):
        super(MyListDense, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x

net = MyListDense()
print(net)


# 使用ParameterDict
class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
                'linear1': nn.Parameter(torch.randn(4, 4)),
                'linear2': nn.Parameter(torch.randn(4, 1))
        })
        # 使用update()新增参数，使用keys()返回所有键值，使用items()返回所有键值对
        self.params.update({'linear3': nn.Parameter(torch.randn(4, 2))})  # 新增

    def forward(self, x, choice='linear1'):
        return torch.mm(x, self.params[choice])

net = MyDictDense()
print(net)

# 样就可以根据传入的键值来进行不同的前向传播：
x = torch.ones(1, 4)
print(net(x, 'linear1'))
print(net(x, 'linear2'))
print(net(x, 'linear3'))

net = nn.Sequential(
    MyDictDense(),
    MyListDense(),
)
print(net)
print(net(x))


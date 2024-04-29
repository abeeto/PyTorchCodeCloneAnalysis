import torch
from torch import nn
#
# # 读写tensor
# # 创建了Tensor变量x，并将其存在文件名同为x.pt的文件里
# x = torch.ones(3)
# torch.save(x, './x.pt')
# # 然后我们将数据从存储的文件读回内存。
# x2 = torch.load('./x.pt')
# print(x2)
#
# # 我们还可以存储一个Tensor列表并读回内存。
# y = torch.zeros(4)
# torch.save([x, y], 'xy.pt')
# xy_list = torch.load('xy.pt')
# print(xy_list)
#
# torch.save({'x': x, 'y': y}, 'xy_dict.pt')
# xy = torch.load('xy_dict.pt')
# print(xy)


# 读写模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


net = MLP()
print(net.state_dict())
# state_dict是一个从参数名称隐射到参数Tesnor的字典对象。

# 注意，只有具有可学习参数的层(卷积层、线性层等)才有state_dict中的条目。
# 优化器(optim)也有一个state_dict，其中包含关于优化器状态以及所使用的超参数的信息。
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(optimizer.state_dict())

# 保存和加载模型

# 仅保存和加载模型参数(state_dict)；
# torch.save(model.state_dict(), PATH)  # 推荐的文件后缀名是pt或pth
#
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))

#  保存和加载整个模型
# torch.save(model, PATH)
# model = torch.load(PATH)

# 采用推荐的方法一来实验一下:
X = torch.randn(2, 3)
Y = net(X)

PATH = "./net.pt"
torch.save(net.state_dict(), PATH)

net2 = MLP()
net2.load_state_dict(torch.load(PATH))
Y2 = net2(X)
print(Y2 == Y)  # net和net2都有同样的模型参数,那么对同一个输入X的计算结果将会是一样的。

import torch
import torch.nn as nn
from torch.nn import functional as F

# 生成500个随机样本，每个样本20个特征
X = torch.rand((500, 20))
# 三分类问题，我们吧0、1、2代指分类结果。 结果为500行1列。
y = torch.randint(low=0, high=3, size=(500, 1), dtype=torch.float32)


class Model(nn.Module):

    def __init__(self, in_features=10, out_features=2):
        """ in_features:    输入该网络的特征数目（即输入层上神经元的数目）
            out_features:   神经网络输出的数目（即输出层上的神经元的数目）
        """
        # 初始化，nn.Module.__init__()
        super(Model, self).__init__()
        # 第二层我们放13个神经元, bias要不要有偏差，有截距
        self.linear1 = nn.Linear(in_features, 13, bias=True)
        self.linear2 = nn.Linear(13, 8, bias=True)
        self.output = nn.Linear(8, out_features)

    # 向前/前向/正向 传播
    def forward(self, x):
        z1 = self.linear1(x)
        sigma1 = torch.relu(z1)
        z2 = self.linear2(sigma1)
        sigma2 = torch.sigmoid(z2)
        z3 = self.output(sigma2)
        sigma3 = F.softmax(z3, dim=1)
        return sigma3


input_ = X.shape[1]
output_ = len(y.unique())

# 实例化神经网络
torch.random.manual_seed(996)
net = Model(in_features=input_, out_features=output_)

res = net.forward(X)

print(res)

# 调用权重矩阵
# print(net.linear1.weight)
# print(net.linear2.weight)
# print(net.output.weight)

# x(500,20) linear1自动将其转化成(20,500), w(13,20) * (20,500) -> (13,500)
print(net.linear1.weight.shape)
# w(8,13) * (13,500) -> (8,500)
print(net.linear2.weight.shape)
# w(3,8) * (8,500) -> (3,500) -> (500,3)
print(net.output.weight.shape)

print(net.linear1.bias.shape)
print(net.linear2.bias.shape)


# 将网络转移到GPU运行
# print(net.cuda())
# 转移到CPU
# print(net.cpu())

# 对神经网络中所有的层,init函数中所有的对象都执行同样的操作
# net.apply()
def initial_0(m):  # 权重初始化为0
    if type(m) == nn.Linear:
        m.weight.data.fill_(0)


# 所有曾上权重初始化为0
net.apply(initial_0)
print("-------------------------------------------")

#
for p in net.parameters():
    print(p)

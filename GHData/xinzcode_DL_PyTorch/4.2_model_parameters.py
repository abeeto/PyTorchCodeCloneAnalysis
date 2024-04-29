import torch
from torch import nn
from torch.nn import init

# 含单隐藏层的多层感知机
net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))  # pytorch已进行默认初始化

print(net)
X = torch.rand(2, 4)
Y = net(X).sum()
print(X, Y)

# 以通过Module类的parameters()或者named_parameters方法来访问所有参数（以迭代器的形式返回），
# 后者除了返回参数Tensor外还会返回其名字。
print(type(net.named_parameters()))
for name, param in net.named_parameters():
    print(name, param.size())

# 再来访问net中单层的参数。通过方括号[]来访问网络的任一层。索引0表示隐藏层为Sequential实例最先添加的层。
for name, param in net[0].named_parameters():
    print(name, param.size(), type(param))


# 返回的param的类型为torch.nn.parameter.Parameter，其实这是Tensor的子类，
# 和Tensor不同的是如果一个Tensor是Parameter，那么它会自动被添加到模型的参数列表里，
class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        self.weight2 = torch.rand(20, 20)
    def forward(self, x):
        pass

n = MyModel()
for name, param in n.named_parameters():
    print(name)

# 因为Parameter是Tensor，即Tensor拥有的属性它都有，
weight_0 = list(net[0].parameters())[0]
print(weight_0.data)
print(weight_0.grad)  # 反向传播前梯度为None
Y.backward()
print(weight_0.grad)

# 初始化模型参数
for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)

for name, param in net.named_parameters():
    if 'bias' in name:
        init.constant_(param, val=0)
        print(name, param.data)


#  PyTorch是怎么实现这些初始化方法的
def normal_(tensor, mean=0, std=1):
    with torch.no_grad():
        return tensor.normal_(mean, std)


# 自定义初始化方法
# 在下面的例子里，令权重有一半概率初始化为0，
# 有另一半概率初始化为[−10,−5][−10,−5]和[5,10][5,10]两个区间里均匀分布的随机数。
def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()

for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)

for name, param in net.named_parameters():
    if 'bias' in name:
        param.data += 1
        print(name, param.data)

# 共享模型参数
linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear)
print(net)
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)
# 在内存中，这两个线性层其实一个对象:
print(id(net[0]) == id(net[1]))
print(id(net[0].weight) == id(net[1].weight))

# 因为模型参数里包含了梯度，所以在反向传播计算时，这些共享的参数的梯度是累加的:
x = torch.ones(1, 1)
y = net(x).sum()
print(x)
print(y)
y.backward()
print(net[0].weight.grad)  # 单次梯度是3，两次所以就是6

import torch
from torch import nn

# 查看GPU是否可用:
print(torch.cuda.is_available())  # 输出 True
# 查看GPU数量：
print(torch.cuda.device_count())  # 输出 1
#  查看当前GPU索引号，索引号从0开始：
print(torch.cuda.current_device())  # 输出 0
# 根据索引号查看GPU名字:
print(torch.cuda.get_device_name(0))


# Tensor的GPU计算
# 默认情况下，Tensor会被存在内存上。因此，之前我们每次打印Tensor的时候看不到GPU相关标识。
x = torch.tensor([1, 2, 3])
print(x)

# 使用.cuda()可以将CPU上的Tensor转换（复制）到GPU上。
# 如果有多块GPU，我们用.cuda(i)来表示第 ii 块GPU及相应的显存（ii从0开始）且cuda(0)和cuda()等价。
x = x.cuda(0)
print(x)
print(x.device)  # 可以通过Tensor的device属性来查看该Tensor所在的设备  输出cuda:0

# 可以直接在创建的时候就指定设备。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.tensor([1, 2, 3], device=device)
# or
# x = torch.tensor([1, 2, 3]).to(device)
print(x)

# 如果对在GPU上的数据进行运算，那么结果还是存放在GPU上。
y = x**2
print(y)

# 需要注意的是，存储在不同位置中的数据是不可以直接进行计算的。
# 即存放在CPU上的数据不可以直接与存放在GPU上的数据进行运算，位于不同GPU上的数据也是不能直接进行计算的。
# z = y + x.cpu()  # 会报错


# 模型的GPU计算
net = nn.Linear(3, 1)
print(list(net.parameters())[0].device)
# 将其转换到GPU上:
net.cuda()
print(list(net.parameters())[0].device)

# 同样,需要保证模型输入的Tensor和模型都在同一设备上，否则会报错。
x = torch.rand(2, 3).cuda()
print(net(x))

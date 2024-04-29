'''
@author: Dzh
@date: 2019/12/3 16:37
@file: torch_regression.py
'''
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 从-1到1均匀生成10个数据
tensor_x = torch.linspace(-1, 1, 100)
'''
这里是把一维矩阵转二维矩阵，dim=1表示从tensor_x.shape的第1个下标维度添加1维
tensor_x.shape 是一维矩阵，大小是10，没有方向
添加后shape变成了(10, 1)
'''
x = torch.unsqueeze(tensor_x, dim=1)
# y = x的平方加上一些噪点, torch.rand()是生成一串指定size，大于等于0小于1的数据
y = x.pow(2) + 0.2*torch.rand(x.size())

# x = Variable(x)
# y = Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

k = True

class Net(torch.nn.Module):
    '''
    这是一个三层的神经网络
    '''
    def __init__(self, n_feature, n_hidden, n_output):
        '''
        初始化
        :param n_feature: 特征数
        :param n_hidden: 神经元隐藏层数
        :param n_output: 输出数
        '''
        super(Net, self).__init__()
        # 参数一是前一层网络神经元的个数，参数二是该网络层神经元的个数
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # relu 激活函数，把小于或等于0的数直接等于0。此时得到第二层的神经网络数据
        x = F.relu(self.hidden(x))
        # 得到第三层神经网络输出数据
        x = self.predict(x)
        return x

# 创建一个三层的神经网络， 每层的神经元数量分别是1， 10 ，1
net = Net(1, 10, 1)

# SGD是一种优化器，net.parameters()是神经网络中的所有参数，并设置学习率
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
# 定义损失函数, MSELoss代表均方差
loss_func = torch.nn.MSELoss()

# 让plt变成实时打印，打开交互模式
plt.ion()
plt.show()

for t in range(100):
    # 调用搭建好的神经网络模型，得到预测值
    prediction = net(x)
    # 用定义好的损失函数，得出预测值和真实值的loss
    loss = loss_func(prediction, y)

    # 每次都需要把梯度将为0
    optimizer.zero_grad()
    # 误差反向传递
    loss.backward()
    # 调用优化器进行优化,将参数更新值施加到 net 的 parameters 上
    optimizer.step()

    if t % 10 == 0:
        # 清除当前座标轴
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        # r- 是红色 lw 是线宽
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        '''
        给图形添加标签，0.5， 0 表示X轴和Y轴坐标，
        'Loss=%.4f'%loss.data.numpy()表示标注的内容，
        .4f表示保留小数点后四位
        fontdict是设置字体大小和颜色
        '''
        plt.text(0.5, 0, 'Loss=%.4f'%loss.data.numpy(), fontdict={'size':20, 'color': 'red'})
        # 间隔多久再次进行绘图
        plt.pause(0.1)

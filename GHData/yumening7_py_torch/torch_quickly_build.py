'''
@author: Dzh
@date: 2019/12/6 17:52
@file: torch_quickly_build.py
'''
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

n_data = torch.ones(100, 2)
# 生成以2为均值，以1为标准差的正态分布数据
x0 = torch.normal(2*n_data, 1)
# 生成与x0对应的分类
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)
'''
torch.cat是拼接两个tensor类型的数据，相当于numpy.concatenate。
参数1是要拼接的数据按元组排列
参数2是拼接方向，0是列，1是按行
type是指定Tensor的类型为Float32bit和Long64bit 
'''
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)

x = Variable(x)
y = Variable(y)
# s是点的大小，lw是线宽，cmap 是指定颜色列表
# plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()


class Net(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        # 参数一是前一层网络神经元的个数，参数二是该网络层神经元的个数
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

# 这里的输入层有2个神经元，隐藏层还是保持10个神经元，输出层有两个神经元
net1 = Net(2, 10, 2)

# 快速搭建法
net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2),
)

print(net1)
print(net2)

# # 让plt变成实时打印，打开交互模式
# plt.ion()
# plt.show()
#
# # SGD是一种优化器，net.parameters()是神经网络中的所有参数，并设置学习率
# optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
# # 定义损失函数类型为交叉熵
# loss_func = torch.nn.CrossEntropyLoss()
#
# for t in range(100):
#     # 调用搭建好的神经网络模型，得到预测值
#     out = net(x)
#     # 用定义好的损失函数，得出预测值和真实值的loss
#     loss = loss_func(out, y)
#     # 每次都需要把梯度将为0
#     optimizer.zero_grad()
#     # 误差反向传递
#     loss.backward()
#     # 调用优化器进行优化,将参数更新值施加到 net 的 parameters 上
#     optimizer.step()
#     if t % 2 ==0:
#         # 清除当前座标轴
#         plt.cla()
#         '''
#         torch.max是查找最大值
#         第一个参数是二维数组，会找出每行或每列的最大值
#         第二个参数为0代表行，1代表列
#         F.softmax(out)是返回所有分类的概率
#         torch.max(F.softmax(out), 1)返回一个长度为2的列表，0下标中是概率矩阵，1下标中是分类矩阵
#         '''
#         prediction = torch.max(F.softmax(out), 1)[1]
#         # 把shapez中为1的维度去掉
#         pred_y = prediction.data.numpy().squeeze()
#         # 真实值y的numpy形式
#         real_y = y.data.numpy()
#         # 绘制散点图
#         plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
#         # 算出分类的准确率
#         accuracy = sum(pred_y == real_y) / 200
#         # 给图形添加标签
#         plt.text(1.5, -4, 'Accuracy=%.2f'%accuracy, fontdict={'size':20, 'color':'red'})
#         # 间隔多久再次进行绘图
#         plt.pause(0.1)
#
# plt.ioff()
# plt.show()
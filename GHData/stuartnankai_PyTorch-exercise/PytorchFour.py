import torch
from torch.autograd import Variable
import torch.nn.functional as F  # get AF # 激励函数都在这
import matplotlib.pyplot as plt


# Method 1
n_data = torch.ones(100, 2)
# print(n_data)
x0 = torch.normal(2 * n_data, 1)

# print(x0)
y0 = torch.zeros(100)
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)

x, y = Variable(x), Variable(y)


# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0)
#
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 继承 __init__ 功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden) # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output) # 输出层线性输出

    def forward(self, x): # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))
        x = self.predict(x) # 输出值, 但是这个不是预测值, 预测值还需要再另外计算

        return x


net = Net(2, 10, 2) # 几个类别就几个 output
print(net)

plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.02) # 传入 net 的所有参数, 学习率
loss_func = torch.nn.CrossEntropyLoss()

for i in range(100):
    out = net(x) # 喂给 net 训练数据 x, 输出分析值

    loss = loss_func(out, y) # 计算两者的误差

    optimizer.zero_grad() # 清空上一步的残余更新参数值
    loss.backward() # 误差反向传播, 计算参数更新值
    optimizer.step() # 将参数更新值施加到 net 的 parameters 上
    if i % 2 == 0:
        plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200.  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.2)

plt.ioff() # 停止画图
plt.show()



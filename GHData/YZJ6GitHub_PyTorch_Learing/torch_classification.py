import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F

# 假数据
n_data = torch.ones(100, 2)         # 数据的基本形态
x0 = torch.normal(2*n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # 类型0 y data (tensor), shape=(100, )
x1 = torch.normal(-2*n_data, 1)     # 类型1 x data (tensor), shape=(100, 1)
y1 = torch.ones(100)                # 类型1 y data (tensor), shape=(100, )

# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
# C = torch.cat( (A,B),0 )  #按维数0拼接（竖着拼）
# C = torch.cat( (A,B),1 )  #按维数1拼接（横着拼）
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1),).type(torch.LongTensor)    # LongTensor = 64-bit integer
# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
print(x.shape)
print(y.shape)

# 画图
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()


class Net(torch.nn.Module): # 继承 torch 的 Module
    def __init__(self,n_feature,n_hidden,n_output):  # 继承 __init__ 功能
        super(Net,self).__init__()
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature,n_hidden) # 隐藏层线性输出
        self.prediect = torch.nn.Linear(n_hidden,n_output) # 输出层线性输出

    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.prediect(x)
        return x
net = Net(n_feature = 2,n_hidden = 10,n_output= 2)
print(net)
# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)# 传入 net 的所有参数, 学习率

loss_func = torch.nn.CrossEntropyLoss()  # 预测值和真实值的误差计算公式 (均方差)

for i in range(100):
    out = net(x)  #  在分类任务中，神经网络输出的结果并非是概率值[10,-2,30]
    #prediction = torch.max(F.softmax(out), 1)[0]
    loss = loss_func(out, y)  # 计算两者的误差
    optimizer.zero_grad()
    loss.backward()  # 误差反向传播, 计算参数更新值
    optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

    if i % 20 == 0:
        plt.cla()
        print("*********************")
        # F.softmax ： 根据输入，将输入转化为概率值，总和为1
        # F.softmax(out,dim = 0) ： 列和为1
        # F.softmax(out,dim = 1) ： 行和为1
        # torch.max(data, dim=0) : 按照列取最大值
        # torch.max(data, dim=1) : 按照行取最大值
        prediction = torch.max(F.softmax(out),1)[1]     # 1 ： 索引值 0 ： 概率值  prediction概率最大值最大值对应的索引值
        #variable ----> numpy()
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        #计算精度
        accuracy = sum(pred_y == target_y) / 200.  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
    plt.ioff()  # 停止画图
    plt.show()







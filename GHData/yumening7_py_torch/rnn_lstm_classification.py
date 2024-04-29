'''
@author: Dzh
@date: 2020/1/8 14:15
@file: rnn_lstm_classification.py
'''

import torch
from torch import nn
import torchvision.datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.autograd import Variable
import time

# 定义超参数
EPOCH = 1                       # 迭代次数
BATCH_SIZE = 64                 # 每一批训练的数据大小
TIME_STEP = 28                  # RNN 考虑了多少个时间点的数据，每一个时间点代表一层神经网络，这里是每一行像素分析做为一个时间点，既图片高度
INPUT_SIZE = 28                 # 每一个时间点会给RNN输入多少个数据点，也就是每一个时间点具备多少个特征，既图片宽度
LR = 0.01                       # 学习率
DOWNLOAD_MNIST = False         # 是否需要下载数据集，首次运行需要把该参数指定为True

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_data  = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
    transform=transforms.ToTensor()
)

'''
非卷积神经网络，不需要额外添加图片高度。
但必须要做特征缩放（这里除255是做了归一化的处理）
这是因为RNN神经网络数据之间具有关联性，如果不做特征缩放，神经网络层之间特征值的巨大差异，
会很大程度影响到彼此的数据输出，无法达到很好的拟合效果。
而CNN卷积神经网络，由于是每一个像素点的R、G、B值或亮度值都会分别对应一个w，所以像素点
之间没有具备关联性，在使用CNN进行图片分类时，不做特征缩放也不会对最后的拟合效果产生影响。
'''
test_x = test_data.data.type(torch.FloatTensor)[:2000]/255

# 获取正确分类的结果
test_y = test_data.targets[:2000]



# 搭建RNN神经网络模型
class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(             # 使用LSTM方式的RNN
            input_size=INPUT_SIZE,      # 输入层的神经元数量
            hidden_size=64,             # 隐藏的神经元数量
            num_layers=1,               # RNN的层数，通常情况下，层数越多，拟合结果越好，但是训练耗时越长
            batch_first=True            # batch是否在size中的第一个维度
        )
        self.out = nn.Linear(64, 10)    # 把经过RNN的数据再放入全连接神经网络

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out 是上一层神经网络的数据(last_hidden_data)与当前层神经网络的数据(current_hidden_data)共同分析得出的数据，r_out shape (batch, time_step, output_size)
        # (h_n, h_c) 是当前层神经网络的数据，由于使用的是LSTM的方式的RNN，所以会得出h_n和h_c两个数据
        # h_n代表“分线剧情”的神经网络数据(branch_hidden_data)，h_n shape (n_layers, batch, hidden_size)
        # h_c代表“主线剧情”的神经网络数据（master_hidden_data），h_c shape (n_layers, batch, hidden_size)
        # self.rnn(x, None) ， None代表最开始是没有上一次神经网络数据的，如果这只是个中间模型，可能还有接入其他模型最后一层的数据做为 last_hidden_data
        r_out, (h_n, h_c) = self.rnn(x, None)
        # 选择最后一层神经网络的输出，传入全连接神经网络
        out = self.out(r_out[:, -1, :])
        return out

rnn = RNN()                                                     # 创建RNN模型对象
print(rnn)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)           # 指定分类器
loss_func = nn.CrossEntropyLoss()                               # 交叉熵的误差计算方式

start_time = time.time()

for epoch in range(EPOCH):
    for setp, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.view(-1, 28, 28)                              # 把b_x的size从([64, 1, 28, 28])， 改变为 ([64, 28, 28])

        output = rnn(b_x)                                       # 传入测试数据，得出预测值
        loss = loss_func(output, b_y)                           # 对比预测值与真实值，得出loss值
        optimizer.zero_grad()                                   # 优化器梯度归零
        loss.backward()                                         # 进行误差反向传递
        optimizer.step()                                        # 将参数更新值施加到 cnn 的 parameters 上

        if setp % 50 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1]               # torch.Size([2000, 10])
            accuracy = (sum(pred_y == test_y)).numpy() / test_y.size(0) # 计算识别精度
            print('Epoch：', epoch, '| train loss：%.4f' % loss.data.numpy(), '| test accuracy：%.2f' % accuracy)

# 使用训练好的模型，预测前十个数据，然后和真实值对比
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1]
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')

print('耗时：', time.time() - start_time)
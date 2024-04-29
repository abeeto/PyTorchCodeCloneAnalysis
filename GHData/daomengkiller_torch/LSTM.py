import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

torch.manual_seed(1)  # 设置随机数种子

EPOCH = 1  # 周期
BATCH_SIZE = 64  # 一次批处理的大小
TIME_STEP = 64  # rnn时间步数/图片高度
INPUT_SIZE = 28  # rnn每步输入值/图片每行像素
LR = 0.01
DOWNLOAD_MNIST = False

train_data = dsets.MNIST(
    root='./mnist/',
    train=True,
    transform=transforms.ToTensor(),  # 将数据转换为张量数据
    download=DOWNLOAD_MNIST,
)

# 显示训练的图标
print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)  # 设置数据加载的方式

# 验证数据集
test_data = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000] / 255.
test_y = test_data.test_labels.numpy().squeeze()[:2000]


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,  # 隐藏层的细胞个数
            num_layers=1,  # 层数
            batch_first=True,  # 约定维数的第一个是批的次数
        )
        self.out = nn.Linear(64, 10)  # 输入64，输出10

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # r_out为输出结果，（h_n,h_c)为主状态，次状态

        out = self.out(r_out[:, -1, :])  # 将最后结果输出
        return out


rnn = RNN()
print(rnn)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # 梯度下降方法
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28, 28))
        b_y = Variable(y)

        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y == test_y) / float(test_y.size)
            print('Epoch:', epoch, '| train loss:%.4f' % loss.data[0], '| test accuracy:%.2f' % accuracy)
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()  # 预测值
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')

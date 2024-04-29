import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.Tensor.unsqueeze(torch.Tensor.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2 * torch.rand()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__() #继承init
        #定义每层的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        #正向传播输入值，神经网络分析出输出值
        x = F.relu(self.hidden(x)) #激励函数（隐藏层的线性值）
        x = self.predict(x) #输出值
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)

print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.2) #传入net的所有参数，学习率
loss_func = torch.nn.MSELoss() #预测值和真实值的误差计算公式（均方差）

for t in range(100):
    prediction = net(x)

    loss = loss_func(prediction, y)

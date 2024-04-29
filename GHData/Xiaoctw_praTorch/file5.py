import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as fun  # 激活函数都在这里
import matplotlib.pyplot as plt


class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):  # 需要传入参数，以此来构建两个层
        super(TwoLayerNet, self).__init__()
        self.Linear1 = nn.Linear(D_in, H)
        self.Linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        # h_relu = self.Linear1(x).clamp(min=0)
        h_relu = fun.relu(self.Linear1(x))
        #两种激活函数的实例
        # h_tanh=fun.tanh(self.Linear1(x))
        # h_sigmod=fun.sigmoid(self.Linear1(x))
        y_pred = self.Linear2(h_relu)
        return y_pred


N, D_in, H, D_out = 64, 1000, 10, 1
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)
model = TwoLayerNet(D_in, H, D_out)
criterion = nn.MSELoss(reduction='sum')
# optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)  # 进行优化,对于这种十分简单的模型，往往SGD效果更好
pre = 100
losses = []
for t in range(3000):
    y_pred = model(x)  # 给模型传入一个输入，得到一个输出
    loss = criterion(y_pred, y)
    print("{}次迭代，loss值为：{}".format(t, loss))
    optimizer.zero_grad()
    loss.backward()
    losses.append(loss)
    pre = loss
    if loss < pre:
        break
    optimizer.step()
plt.plot(losses)
plt.show()

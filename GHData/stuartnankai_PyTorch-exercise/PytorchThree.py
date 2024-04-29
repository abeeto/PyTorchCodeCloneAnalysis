import torchvision
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

# AF part
# x = torch.linspace(-5,5,200)
# x = Variable(x)
# x_np = x.data.numpy()
# y_relu = F.relu(x).data.numpy()
# y_sigmoid = torch.sigmoid(x).data.numpy()
# y_tanh =torch.tanh(x).data.numpy()
# y_softplus = F.softplus(x).data.numpy()
#
#
# plt.figure(1,figsize=(8,6))
# plt.subplot(221)
# plt.plot(x_np,y_relu,c='red',label = 'relu')
# plt.ylim(-1,5)
# plt.legend(loc='best')
#
# plt.subplot(222)
# plt.plot(x_np,y_sigmoid,c='red',label = 'sigmoid')
# plt.ylim(-0.2,1.2)
# plt.legend(loc='best')
#
#
# plt.subplot(223)
# plt.plot(x_np,y_tanh,c='red',label = 'tanh')
# plt.ylim(-1.2,1.2)
# plt.legend(loc='best')
#
# plt.subplot(224)
# plt.plot(x_np,y_softplus,c='red',label = 'softplus')
# plt.ylim(-0.2,6)
# plt.legend(loc='best')
#
# plt.show()

# numpy and torch
#
# np_data = np.arange(6).reshape((2,3))
# torch_data = torch.from_numpy(np_data)
# tensor2arry = torch_data.numpy()
#
# print(
#     '\n numpy', np_data,
#     '\n torch', torch_data,
#     '\n tensorarray', tensor2arry,
# )

# abs

# absdata = [-1,-2]
# tensor = torch.FloatTensor(data)
# print(torch.mean(tensor))


# data = [[1,2],[3,4]]
#
# tensor = torch.FloatTensor(data)
#
# print(
#     '\n numpy', np.matmul(data,data),
#     '\n torch', torch.mm(tensor,tensor)
# )

# tensor = torch.FloatTensor([[1,2],[3,4]])
# variable = Variable(tensor,requires_grad = True)
# t_out = torch.mean(tensor*tensor)
# v_out = torch.mean(variable*variable)
#
# v_out.backward()
# print(variable.grad)
# print(variable.data.numpy())


# Regression model

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # from 1d to 2d
y = x.pow(2) + 0.2 * torch.rand(x.size())  # add noise
# y = x.pow(2)


x, y = Variable(x), Variable(y)


#
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):  # 定义 组成部分
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

        pass

    def forward(self, x):  # 搭建过程
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(1, 10, 1)
print(net)

plt.ion()  # 实时打印
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()

for i in range(100):
    prediction = net(x)

    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

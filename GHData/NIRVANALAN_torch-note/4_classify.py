import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 假数据
n_data = torch.ones(100, 2)  # 数据的基本形态
x0 = torch.normal(2 * n_data, 1)  # 类型0 x data (tensor), shape=(100, 2) 标准差为1， 均值为2*n_data中的元素(2)的正态分布中随机生成的
y0 = torch.zeros(100)  # 类型0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2 * n_data, 1)  # 类型1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)  # 类型1 y data (tensor), shape=(100, 1)

# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)  # LongTensor = 64-bit integer

# torch 只能在 Variable 上训练, 所以把它们变成 Variable
x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')

# plt.show()

# class Net(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
#         self.out = torch.nn.Linear(n_hidden, n_output)   # output layer
#
#     def forward(self, x):
#         x = F.relu(self.hidden(x))      # activation function for hidden layer
#         x = self.out(x)
#         return x
#
# net = Net(n_feature=2, n_hidden=10, n_output=2)     # define the network
net = torch.nn.Sequential(
	torch.nn.Linear(2, 10),
	torch.nn.ReLU(),
	torch.nn.Linear(10, 2)
)
print(net)  # net architecture

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

plt.ion()  # something about plotting

for t in range(100):
	out = net(x)  # input x and predict based on x
	loss = loss_func(out, y)  # must be (1. nn output, 2. target), the target label is NOT one-hotted
	
	optimizer.zero_grad()  # clear gradients for next train
	loss.backward()  # back propagation, compute gradients
	optimizer.step()  # apply gradients
	
	if t % 2 == 0:
		# plot and show learning process
		print(t)
		plt.cla()
		prediction = torch.max(out, 1)[1]
		pred_y = prediction.data.numpy().squeeze()
		target_y = y.data.numpy()
		plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
		accuracy = sum(pred_y == target_y) / 200.
		plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
		plt.pause(0.1)
		

plt.ioff()
plt.show()

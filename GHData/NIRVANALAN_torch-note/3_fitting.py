import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import time

t1 = time.time()
device = torch.device('cuda')
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1).cuda()  # transfer to [100,1]
y = x.pow(2) + 0.2 * torch.rand(x.size()).cuda()
x, y = Variable(x), Variable(y)

plt.scatter(x.data.cpu().numpy(), y.data.cpu().numpy())
plt.show()


# nn
class Net(torch.nn.Module):
	def __init__(self, n_feature, n_hidden, n_output):
		super().__init__()
		# define layer
		self.hidden = nn.Linear(n_feature, n_hidden).cuda()
		self.predict = nn.Linear(n_hidden, n_output).cuda()
	
	def forward(self, x):  # forward function in Module
		x = F.relu(self.hidden(x))
		x = self.predict(x)
		return x


net = Net(n_feature=1, n_hidden=20, n_output=1)
print(net)

# optimizer and loss func
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
loss_func = nn.MSELoss()  # mean square error

plt.ion()

for t in range(500):
	prediction = net(x)
	loss = loss_func(prediction, y)
	
	optimizer.zero_grad()
	loss.backward()  # calculate the updated value of params
	optimizer.step()  # update the params
	
	if t % 5 == 0:
		plt.cla()
		plt.scatter(x.data.cpu().numpy(), y.data.cpu().numpy())
		plt.plot(x.data.cpu().numpy(), prediction.data.cpu().numpy(), 'r-', lw=5)
		plt.text(0.5, 0, 'loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
		plt.pause(0.1)

plt.ioff()
plt.show()
print(time.time() - t1)

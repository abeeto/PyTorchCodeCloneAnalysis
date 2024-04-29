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


def save():
	# nn
	net1 = nn.Sequential(
		nn.Linear(1, 10),
		nn.ReLU(),
		nn.Linear(10, 1)
	).cuda()
	
	# optimizer and loss func
	optimizer = torch.optim.Adam(net1.parameters(), lr=1e-2)
	loss_func = nn.MSELoss()  # mean square error
	# plt.ion()
	for t in range(500):
		prediction = net1(x)
		loss = loss_func(prediction, y)
		optimizer.zero_grad()
		loss.backward()  # calculate the updated value of params
		optimizer.step()  # update the params
	
	# if t % 5 == 0:
	# 	plt.cla()
	# 	plt.scatter(x.data.cpu().numpy(), y.data.cpu().numpy())
	# 	plt.plot(x.data.cpu().numpy(), prediction.data.cpu().numpy(), 'r-', lw=5)
	# 	plt.text(0.5, 0, 'loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
	# 	plt.pause(0.1)
	
	plt.figure(1, figsize=(10, 3))
	plot('net1', 131, prediction)
	
	torch.save(net1, 'net.pkl')
	torch.save(net1.state_dict(), 'net_params.pkl')  # save the params


# plt.ioff()
# plt.show()
# print(time.time() - t1)

def restore_net():
	net2 = torch.load('net.pkl')
	prediction = net2(x)
	plot('net2', 132, prediction)


def restore_params():
	net3 = nn.Sequential(
		nn.Linear(1, 10),
		nn.ReLU(),
		nn.Linear(10, 1)
	).cuda()
	net3.load_state_dict(torch.load('net_params.pkl'))
	plot('net3', 133, prediction=net3(x))


def plot(name, num, prediction):
	plt.subplot(num)
	plt.title(name)
	plt.scatter(x.data.cpu().numpy(), y.data.cpu().numpy())
	plt.plot(x.data.cpu().numpy(), prediction.data.cpu().numpy(), 'r-', lw=5)


if __name__ == '__main__':
	save()
	
	restore_net()
	
	restore_params()

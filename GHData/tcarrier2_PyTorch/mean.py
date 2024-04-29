import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

N, D_in, H, D_out = 10000, 14, 14, 2
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

for i in range(0,N):
	c_x = 0
	c_y = 0
	for j in range(0,D_in):
		if j%2==0:
			c_x = c_x + x[i,j]
		else:
			c_y = c_y + x[i,j]
	y[i,0] = c_x / (D_in / 2.0)
	y[i,1] = c_y / (D_in / 2.0)

model = torch.nn.Sequential(
	torch.nn.Linear(D_in, H),
	################torch.nn.Sigmoid(),
	torch.nn.Linear(H, D_out))
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
for t in range(500):
	y_pred = model(x)
	loss = loss_fn(y_pred,y)

	model.zero_grad()
	loss.backward()

	for param in model.parameters():
		param.data -= learning_rate*param.grad.data

x_TEST = Variable(torch.randn(1, D_in))
y_TEST = model(x_TEST)

c_x = 0.0
c_y = 0.0
for i in range(0, int(D_in/2.0)):
	plt.plot([x_TEST[0,2*i]], [x_TEST[0,2*i+1]], 'go')
	c_x = c_x + x_TEST[0,2*i]
	c_y = c_y + x_TEST[0,2*i+1]
	
plt.plot([y_TEST[0,0]], [y_TEST[0,1]], 'ro')
plt.plot([c_x/(D_in/2.0)], [c_y/(D_in/2.0)], 'bx')
plt.axis([-7, 7, -7, 7])
plt.show()


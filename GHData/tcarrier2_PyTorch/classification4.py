import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

N, D_in, H, D_out = 4000, 2, 11, 3
x = Variable(torch.randn(N, D_in))
y = Variable(torch.LongTensor(N), requires_grad=False)

for i in range(0,N):
	if x[i,1] > x[i,0]**2:
		y[i] = 0
	elif x[i,1] < -x[i,0]**2:
		y[i] = 1
	else:
		y[i] = 2

model = torch.nn.Sequential(
	torch.nn.Linear(D_in, H, bias=True),
	torch.nn.ReLU(),
	torch.nn.Linear(H, D_out, bias=True))
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(1000):
	y_pred = model(x)	
	loss = loss_fn(y_pred,y)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

N_TEST = 500
x_TEST = Variable(torch.randn(N_TEST, D_in))
y_TEST = Variable(torch.LongTensor(N_TEST), requires_grad=False)
y_TEST = model(x_TEST)

colors = ['yo','bo','go']

for i in range(0,N_TEST):
	temp = y_TEST[i,:].cpu().detach().numpy()
	index_max = np.argmax(temp)
	plt.plot([x_TEST[i,0]], [x_TEST[i,1]], colors[index_max])

eq_x = np.linspace(-10, 10, 1000)
eq_y = eq_x**2
plt.plot(eq_x,eq_y)
eq_y = -eq_x**2
plt.plot(eq_x,eq_y)
plt.axis([-4, 4, -4, 4])
plt.show()


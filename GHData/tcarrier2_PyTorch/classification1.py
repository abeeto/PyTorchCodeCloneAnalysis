import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

N, D_in, H, D_out = 4000, 2, 11, 1
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)
radius = 0.5

for i in range(0,N):
	if x[i,0]**2 + x[i,1]**2 < radius**2:
		y[i,0] = 0
	else:
		y[i,0] = 1

model = torch.nn.Sequential(
	torch.nn.Linear(D_in, H, bias=True),
	torch.nn.ELU(),
	torch.nn.Linear(H, D_out, bias=True),
	torch.nn.Sigmoid())
loss_fn = torch.nn.BCELoss()
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
y_TEST = model(x_TEST)

for i in range(0,N_TEST):
	if y_TEST[i,0] < 0.5:
		plt.plot([x_TEST[i,0]], [x_TEST[i,1]], 'ro')
	else:
		plt.plot([x_TEST[i,0]], [x_TEST[i,1]], 'bo')

eq_x = np.linspace(-10, 10, 1000)
eq_y = radius*np.sin(eq_x)
eq_x = radius*np.cos(eq_x)
plt.plot(eq_x,eq_y)
plt.axis([-4, 4, -4, 4])
plt.show()


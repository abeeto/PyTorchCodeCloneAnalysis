import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

N, D_in, H, D_out = 4000, 1, 31, 1
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

for i in range(0,N):
	y[i,0] = x[i,0]**2

model = torch.nn.Sequential(
	torch.nn.Linear(D_in, H),
	torch.nn.ReLU(),
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

N_TEST = 100
x_TEST = Variable(torch.randn(N_TEST, D_in))
y_TEST = model(x_TEST)

for i in range(0,N_TEST):
	plt.plot([x_TEST[i,0]], [y_TEST[i,0]], 'go')

eq_x = np.linspace(-10, 10, 1000)
eq_y = eq_x**2
plt.plot(eq_x,eq_y)
plt.axis([-4, 4, -1, 4])
plt.show()


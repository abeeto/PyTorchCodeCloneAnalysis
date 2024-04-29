import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('diabetes.csv')
data = data.values.astype(np.float32)

x_data = Variable(torch.from_numpy(data[:,0:-1]))
y_data = Variable(torch.from_numpy(data[:,[-1]]))

class Model(torch.nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.l1 = torch.nn.Linear(8, 6)
		self.l2 = torch.nn.Linear(6, 4)
		self.l3 = torch.nn.Linear(4, 1)
		self.sigmoid = torch.nn.Sigmoid()
		self.relu = torch.nn.ReLU()

	def forward(self, x):
		out1 = self.relu(self.l1(x))
		out2 = self.relu(self.l2(out1))
		y_pred = self.sigmoid(self.l3(out2))
		return y_pred

model = Model()

criterion = torch.nn.BCELoss(size_average=True)

optimizer = torch.optim.Adadelta(model.parameters(), lr=0.019)

for epoch in range(2000):
	y_pred = model(x_data)

	loss = criterion(y_pred, y_data)
	print(epoch, loss.data[0])

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

y_pred = model.forward(x_data).data > 0.5
#s = torch.mean(y_pred)
#print s
#a = (y_pred==y_data)
a = y_pred.numpy()
b = y_data.data.numpy()

print np.sum(a==b)/float(768)
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

class DiabetesDataset(Dataset):
	def __init__(self):
		data = pd.read_csv('diabetes.csv')
		data = data.values.astype(np.float32)
		self.x_data = torch.from_numpy(data[:, 0:-1])
		self.y_data = torch.from_numpy(data[:, [-1]])
		self.len = data.shape[0]
	
	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.len

dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)


class Model(torch.nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.l1 = torch.nn.Linear(8, 5)
		self.l2 = torch.nn.Linear(5, 3)
		self.l3 = torch.nn.Linear(3, 1)

		self.sigmoid = torch.nn.Sigmoid()
		self.relu = torch.nn.ReLU()

	def forward(self, x):
		out1 = self.relu(self.l1(x))
		out2 = self.relu(self.l2(out1))
		y_pred = self.sigmoid(self.l3(out2))
		return y_pred

model = Model()

criterion = torch.nn.BCELoss(size_average=True)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

iter_list = list()
loss_list = list()
iter = 1

for epoch in range(2):
	for i, data in enumerate(train_loader, 0):
		inputs, labels = data
		inputs, labels = Variable(inputs), Variable(labels)

		y_pred = model(inputs)

		loss = criterion(y_pred, labels)
		loss_list.append(loss.data[0])
		iter_list.append(iter)
		print(epoch, loss.data[0])

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		iter += 1

plt.plot(iter_list, loss_list)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
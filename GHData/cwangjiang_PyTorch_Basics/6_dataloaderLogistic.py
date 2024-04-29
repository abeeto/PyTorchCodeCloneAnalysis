import torch
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F 

# construct dataset
class DiabetesDataset(Dataset):

	def __init__(self):
		xy = np.loadtxt('./data/diabetes.csv.gz', delimiter=',',dtype=np.float32)
		self.len = xy.shape[0]
		self.x_data = torch.from_numpy(xy[:,0:-1])
		self.y_data = torch.from_numpy(xy[:,[-1]])

	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.len

# create instance and train_loader
dataset = DiabetesDataset()
train_loader = DataLoader(dataset = dataset, batch_size = 32, shuffle = True, num_workers = 1)

# Step 1, create NN
class Model(torch.nn.Module):

	def __init__(self):
		super(Model, self).__init__()
		self.l1 = torch.nn.Linear(8,6)
		self.l2 = torch.nn.Linear(6,4)
		self.l3 = torch.nn.Linear(4,1)

		#self.sigmoid = torch.nn.Sigmoid()

	def forward(self, x):
		# out1 = self.sigmoid(self.l1(x))
		# out2 = self.sigmoid(self.l2(out1))
		# y_pred = self.sigmoid(self.l3(out2))
		out1 = F.sigmoid(self.l1(x))
		out2 = F.sigmoid(self.l2(out1))
		y_pred = F.sigmoid(self.l3(out2))
		return y_pred

model = Model()

# Create loss and optimizer
criterion = torch.nn.BCELoss(size_average = True)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

# Training loop
for epoch in range(100):
	for i, data in enumerate(train_loader, 0):
		inputs, labels = data

		y_pred = model(inputs)

		loss = criterion(y_pred, labels)
		print(epoch, i, loss.data[0])

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()







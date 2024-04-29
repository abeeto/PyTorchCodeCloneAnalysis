# This code exted simple NN to multilayer and deep NN, on external data point on a logistic regression task.

import torch
import numpy as np
import torch.nn.functional as F 

# load data 
xy = np.loadtxt('./data/diabetes.csv.gz', delimiter=',',dtype=np.float32)
x_data = torch.from_numpy(xy[:,0:-1])
y_data = torch.from_numpy(xy[:,[-1]])

print(x_data.data.shape)
print(y_data.data.shape)

# Step 1, define multi layer and forwarding function. We can define a sigmoid layer, and apply this layer to regular linear layer
# we can also apply F.sigmoid function directly to the regular linear layers.
class Model(torch.nn.Module):

	def __init__(self):
		super(Model, self).__init__()
		self.l1 = torch.nn.Linear(8,6)
		self.l2 = torch.nn.Linear(6,4)
		self.l3 = torch.nn.Linear(4,1)

		#self.sigmoid = torch.nn.Sigmoid() # sigmoid layer

	def forward(self, x):
		# out1 = self.sigmoid(self.l1(x))
		# out2 = self.sigmoid(self.l2(out1))
		# y_pred = self.sigmoid(self.l3(out2))
		out1 = F.sigmoid(self.l1(x))
		out2 = F.sigmoid(self.l2(out1))
		y_pred = F.sigmoid(self.l3(out2))
		return y_pred

model = Model()

# Step 2, define loss and optimizer
criterion = torch.nn.BCELoss(size_average = True)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1) # SGD update NN for each example in the batch, but still loop over all example? or just use one example?


# Step 3, training
for epoch in range(1000): 
# there are 1000 total loops over all data, each epoch we use all data. 
# if there are N training example, batch size is N, and there are 1 batch, batch number is 1. But since 
# we are using SGD, for each batch (N example) we are still do bp and updating for each example. which equivalent
# we have batch size of 1 and N batches.
	y_pred = model(x_data)

	loss = criterion(y_pred, y_data)
	print(epoch, loss.data[0])

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()




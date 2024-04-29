# Similar code for logistic regression

import torch
import torch.nn.functional as F 

x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0], [1.0]])



# Step 1. define network parameters and forward, it's still linear layer, but for regular NN logistic reguression, there is activation function, 
# can be sigmoid, Relu, tanh..., they are just the function in forward, rather the layer structure. Previous linear regression, there is no
# activation, only direct calculation.
class Model(torch.nn.Module):

	def __init__(self):
		super(Model, self).__init__()
		self.linear = torch.nn.Linear(1,1)

	def forward(self, x):
		y_pred = F.sigmoid(self.linear(x)) 
		return y_pred

model = Model()


# Step 2. define loss and optimizer.
criterion = torch.nn.BCELoss(size_average = True) # sigmoid activation is usually combined with Entropy loss 
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# Step 3. Training
for epoch in range(1000):
	y_pred = model(x_data)

	loss = criterion(y_pred, y_data)
	print(epoch, loss.data[0])

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

# Test
t = torch.tensor([[1.0]])
print("predic 1", 1.0, model(t).data[0][0].item()>0.5)
t = torch.tensor([[7.0]])
print("predic 7", 7.0, model(t).data[0][0].item()>0.5)

# This code is to use pytorch building functions to construct NN, to do linear regression, the task is simple,
# but the basic steps are there, and can be extend to more complicated tasks.
# Three steps:
# 1. Define modle class and parameters and forward
# 2. Construct loss and optimizer
# 3. Training, forward, backward, updating
import torch

 # regular list, which represent the original data, data must have one more dimension to represent each example. Then is the correct dimension for each data, which is 1 here.
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])


# Defind the nn, the parameters for each layer, and the forward functions. similar to the definition of w and forward(x)
class Model(torch.nn.Module):

	def __init__(self):
		super(Model, self).__init__()
		self.linear = torch.nn.Linear(1,1) # linear layer has only two index: input and output dimension, left side is the name of this layer as a function
		# right side defines this layer, and it's parameters.

	def forward(self, x):
		y_pred = self.linear(x)
		return y_pred

# Create an instance, a real object which can be used from class.
model = Model()


# Define criterion for computing loss, and how/what do we update the model, we update "model" using SGD method
criterion = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) # there are other optimizer: Adam, LBFGS, RMSprop, Rprop...


# Training
for epoch in range(500): # 500 total loops

	y_pred = model(x_data)

	loss = criterion(y_pred, y_data)
	print(epoch, loss.data[0])

	optimizer.zero_grad() # clear previous gradients
	loss.backward() # compute current gradient
	optimizer.step() # update all parameters for model.parameters

t = torch.tensor([4.0]) # test data
y_pred = model(t)
print("predict (after training)", 4, model(t).data[0])
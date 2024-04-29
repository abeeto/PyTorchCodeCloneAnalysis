import torch
from torch.autograd import Variable

#Create input and output data in form of tensor in Variable
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))


#Model class is created based on the structure of model
class Model(torch.nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.linear = torch.nn.Linear(1, 1)

	def forward(self, x):
		y_pred = self.linear(x)
		return y_pred

#Instance of model is created
model = Model()

#'criterion' is created for calculating MSE loss
criterion = torch.nn.MSELoss(size_average=False)

#'optimizer' is created for optimizing the model using Stochastic Gradient Descent
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
	y_pred = model(x_data)				#input data is inserted into model of instance

	loss = criterion(y_pred, y_data)	#loss is calculated
	print(epoch, loss.data[0])

	optimizer.zero_grad()				#optimizer is brought down to zero
	loss.backward()						#creates compuatation graph for loss
	optimizer.step()					#It is used to optimize the loss function by updating the value of 'w' using gradient


#Testing time
#Input for test is created using Variable as torch Tensor
x_test = Variable(torch.Tensor([[5.0], [10.0], [9.0], [7.0], [1000.0]]))
	
#Test data is inserted into forward pass of the model to find the predicted output
y_test_pred = model.forward(x_test).data
print("Predicted output for test data is {}".format(y_test_pred))
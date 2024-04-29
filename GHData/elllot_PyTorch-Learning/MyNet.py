import torch
from torch import autograd, nn, optim
import torch.nn.functional as F

"""
Simple 2-linear-layer Neural Net using PyTorch

Was insightful in learning how to instantiate layers as well as getting familiar
with using an optimizer to perfrom gradient descent.

"""
class MyNet(nn.Module): # derives from nn 

	def __init__(self, input_size, hidden_size, num_classes):
		# initialize parent class as this is a derived class
		super().__init__()
		self.h1 = nn.Linear(input_size, hidden_size)
		# second layer output needs to match number of classes
		# since our network only has 2 layers
		self.h2 = nn.Linear(hidden_size, num_classes) 

	def forward(self, input):
		input = self.h1(input) # feed to first layer
		input = F.tanh(input) 
		input = self.h2(input) # feed to second layer
		input = F.softmax(input) # using softmax to get probabilities
		return input

if __name__ == '__main__':

	batch_size = 5 # batch size
	input_size = 4 # input (feature) size
	hidden_size = 4 # hidden layer size
	num_classes = 4 # number of classes in classifier
	learning_rate = 0.001 # learning rate for training
	iterations = 1000 # number of iterations in learning
	seed = 123

	if seed:
		torch.manual_seed(seed)
	input = autograd.Variable(torch.rand(batch_size, input_size)) - 0.5

	# long tensor with random numbers that will serve as our arbitrary target
	target = autograd.Variable((torch.rand(batch_size) * num_classes).long())

	# create model using a Neural Net with defined contraints
	model = MyNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

	# instantiate the optimizer
	optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

	# training 
	for _ in range(iterations):

		# calculate output given our current Nueral Net structure
		output = model(input) 

		# grab our prediction
		_, pred = output.max(1)

		# print helpers. flip matrices horizontally for clearer visualization
		print('target:', str(target.view(1, -1)).split('\n')[1]) 
		print('prediction:', str(pred.view(1, -1)).split('\n')[1])

		# calculate loss with calculated output against target values
		loss = F.nll_loss(output, target) 
		print('loss:', loss[0])

		model.zero_grad()
		loss.backward() # back propagation
		"""
		params - weights to be updated to reduce loss
		deriving off of the module allows us to use .parameters() to access all modules 
		assigned to 'self'
		"""

		# make optimizer take a step into the direction that reduces loss
		optimizer.step()
		
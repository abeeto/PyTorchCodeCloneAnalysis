"""
Here is a linear regression using pytorch. Easy and fun:
"""

import numpy as np 
import torch
from torch.autograd import Variable

#to simulate some dataset, fix the number of observations and inputs
n, p = 100, 5

x = np.random.normal(0, 1, size = (n, p))

true_w = np.random.uniform(.5, 3.5, size = (p, 1))

t = np.dot(x, true_w)

#To use pytorch, we need to transform the numpy arrays to pytorch tensors. Here we have matrices.
#We use torch.from_numpy(<numpy array>) to transform.
#Very important is the .float() part. 
#Since in our optimization, we don't want to update input and target, we set requires_grad = False. 
#In neural networks for the hidden layers, or if weights are initialized we should requires_grad = True.
x_data = Variable(torch.from_numpy(x).float(), requires_grad = False)
t_data = Variable(torch.from_numpy(t).float(), requires_grad = False)


#To use pytorch, we need to define a class (Google "inheritance in python" for more info.)
#Here, the parent class is torch.nn.Mudule
class model(torch.nn.Module):
	def __init__(self):
		#inherite arguments from the parent class:
		super(model, self).__init__()
		#For linear combination of inputs (X*w), use torch.nn.Linear...To see the options run print(dir(torch.nn))
		self.linear = torch.nn.Linear(p, 1)#here the intercept/bias is used (defualt)

		#The forward pass is for prediction of test data. The forward pass for the training is inside the parent class...
	def forward(self, x):
		y = self.linear(x)
		return(y)


#Here we initiate our model. 
my_model = model()

#We define the loss function as follows. Note that the below is a function and need to get arguments: target, and predicted values.
loss = torch.nn.MSELoss()#(size_average = False)

#We need to select an optimization function (it'll get arguments in the loop). Here is SGD. 
#Check documentations for other methods.
#Note the my_model.parameters(). We didn't define any parameter(s). torch.nn.Linear automatically defines and 
#initialize parameters and now we can extract them using my_model.parameters()
#Also note that the type of containers of parameters is torch tensors too...
optimizer  = torch.optim.SGD(my_model.parameters(), lr = 0.01)

#Here we iteratively upadte the weights. We can fix a tolerance and use while loop. But let's keep it simple here.
for epoch in range(100):
	#In the loop, three things should be specified: forward, loss, backward:
	#Here we claculate the updated forward. Note that the instance of class "model" calls torch.nn.Linear and calculate 
	#the forward.
	y = my_model(x_data)

	#loss function takes the target and predicted values. Note that the type of arguments is pytorch tensors...
	l = loss(y, t_data)

	#For the optimization step, first: we clear the updated parameters.
	optimizer.zero_grad()
	#Second: use the backward() method to automatically create the computation graph and evaluate the gradients backward, similar to backpropagation alg.
	l.backward()
	#Use the step() method to perform the update. Note that by specifying the arguments of the optimization method above
	#we can use different update procedures such as adam, momentum, and so on.
	#Also, note that there is not need to define/call the parameters here. Just write the lines of codes in order
	optimizer.step()


#To check visually how the observed and predicted values are close. THe test data is taken from the training data.
x_test = Variable(torch.from_numpy(x[0, :]).float())
y_test = Variable(torch.from_numpy(t[0]).float())

print(my_model.forward(x_test))
print(y_test)


#To check visually how the observed and predicted values are close. THe test data is independet of the training data.
xx = np.random.normal(0, 1, size = (1, p))
x_test = Variable(torch.from_numpy(xx).float())
y_test = Variable(torch.from_numpy(np.dot(xx, true_w)).float())

#Now we use the forward method we defined our model class.
print(my_model.forward(x_test))
print(y_test)

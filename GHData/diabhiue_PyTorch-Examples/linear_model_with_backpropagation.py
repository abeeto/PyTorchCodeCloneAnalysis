#Import numpy, matplotlib.pyplot, torch and torch.autograd.Variable
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.autograd import Variable

#Initialize value of 'w' and 'alpha'
w = Variable(torch.Tensor([5000]), requires_grad=True)	#'w' is declared as torch Variable for using autograd feature of torch
alpha = 0.01

#Input and output data denoted as x_data and y_data respectively
x_data = np.array([1.0, 2.0, 3.0])
y_data = np.array([2.0, 4.0, 6.0])

#Forward pass function of linear model: Y = W*X
def forward(x):
	return w*x

#Calculates loss using mean square error
def loss(x,y):
	y_pred = forward(x)
	return (y_pred-y)**2

#Creates w_list and epoch_list and initialize it
w_list = list([w.data[0]])
epoch_list = list([0])

#intialize epoch as 1
epoch = 1

temp = str('inf')	#stores previous value of 'w' and it is initialized by infinity

#Iterates over each input and output data each epoch till difference between value of consecutive 'w's is less than 10^-18
while (temp > w.data[0]):
	temp = w.data[0]
	for (x,y) in zip(x_data, y_data):
		l = loss(x,y)				#calculate loss
		l.backward()				#creates computation graph for loss
		grad = w.grad.data 			#calculates the gradient of loss with respect to 'w'
		w.data -= alpha*grad 		#update 'w'
		w.grad.data.zero_()			#Gradient is again dropped down to zeros so that it does not interfere with other future gradients
	w_list.append(w.data[0])		#appends 'w' in each epoch
	epoch += 1						#increment epoch no.
	epoch_list.append(epoch+1)		#appends 'epoch' in each epoch
	print("Epoch no. {}, Loss = {}".format(epoch, l))
print w.data[0]
#Plot 'w' vs 'Epochs'
plt.plot(epoch_list, w_list)
plt.xlabel('Epochs')
plt.ylabel('w')
plt.show()
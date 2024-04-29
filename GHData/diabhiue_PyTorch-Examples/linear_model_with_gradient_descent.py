#Import numpy and matplotlib.pyplot
import numpy as np
from matplotlib import pyplot as plt


#Initialize value of 'w' and 'alpha'
w = 5000
alpha = 0.01

#Input and output data denoted as x_data and y_data respectively
x_data = np.array([1.0, 2.0, 3.0])
y_data = np.array([2.0, 4.0, 6.0])

#Forward pass function of linear model: Y = W*X
def forward(x):
	return w*x

#Calculates gradient of the loss function. Here gradient =  2*x*(y_pred - y)
def gradient(y,x):
	return 2*x*(forward(x) - y)

#Calculates loss using mean square error
def loss(x,y):
	y_pred = forward(x)
	return (y_pred-y)**2

#Creates w_list and epoch_list and initialize it
w_list = list([w])
epoch_list = list([0])

#Iterates over each input and output data each epoch
for epoch in range(40):
	for (x,y) in zip(x_data, y_data):
		grad = gradient(y, x)	#calcuate gradient on current data
		w -= alpha*grad 		#update 'w'
		l = loss(x,y)			#calculate loss
	w_list.append(w)			#appends 'w' in each epoch
	epoch_list.append(epoch+1)	#appends 'epoch' in each epoch
	print("Epoch no. {}, Loss = {}".format(epoch, l))

#Plot 'w' vs 'Epochs'
plt.plot(epoch_list, w_list)
plt.xlabel('Epochs')
plt.ylabel('w')
plt.show()
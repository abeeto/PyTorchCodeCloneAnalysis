# This code is to use auto grad to conduct simple linear regression

import torch

 # regular list, which represent the original data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# the parameter for the NN, it's simple, only one parameter is enough
w = torch.tensor([1.0], requires_grad = True) 

# Strucuture of the NN, just one multiplication
def forward(x):
	return x * w
 
# Loss function for one data
def loss(x, y):
	y_pred =  forward(x)
	return (y_pred - y)*(y_pred-y)

# Training
for epoch in range(10): # one epoch is a loop over all data, within one epoch, should be the loop over many mini baches, each mini bach is a subset of all data
	for i in range(len(x_data)): # here each mini bach is one data pair
		x_val = x_data[i]; # this for loop can also be like this: for x_val, y_val in zip(x_data, y_data):
		y_val = y_data[i];
		l = loss(x_val, y_val)
		l.backward()
		print("\tgrad: ", x_val, y_val, w.grad.data[0]) # print gradient for each mini bach
		w.data = w.data - 0.01*w.grad.data # update w for this data pair/mini bach

		w.grad.data.zero_() #clear gradient

	print("progress:", epoch, l.data[0]) # print loss for the last mini bach after each epoch


print("predict (after training)", 4, forward(4).data[0]) # print the prediction for one new input
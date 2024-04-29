#Import numpy and matplotlib.pyplot
import numpy as np 		
from matplotlib import pyplot as plt

#Input and output data denoted as x_data and y_data respectively
x_data = np.array([[1.0, 2.0, 3.0]])
y_data = np.array([[2.0, 4.0, 6.0]])

#Different values of weights of linear model to try on so that we can find the best 'w' on which loss is minimum
W = np.arange(0.0, 4.1, 0.1)
W = W.reshape(W.shape[0], 1)

#Forward pass function of linear model: Y = W*X
def forward(x):
	return W.dot(x)

#Finds loss by performing mean square error(MSE): loss = sum((y-y_pred)^2)
def loss(y_pred, y):
	return np.mean(np.square(y_pred-y), axis=1)

#Predicts the output by using best 'w' among list of 'W'
def predict(w, x_test):
	return w*x_test

#Forward pass and calculates loss on given data
y_pred = forward(x_data)
loss_mse = loss(y_pred, y_data)

#Plot Weight vs Loss for different variants of weights
plt.plot(W, loss_mse)
plt.xlabel('W')
plt.ylabel('Loss')
plt.show()

#Finds best w
w = W[loss_mse == np.min(loss_mse)][0]

#Generates test input data 
x_test = np.arange(0, 201)
#Predicts the correct output
y_test_pred = predict(w, x_test)

#Plots for different values of X
plt.plot(x_test, y_test_pred)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
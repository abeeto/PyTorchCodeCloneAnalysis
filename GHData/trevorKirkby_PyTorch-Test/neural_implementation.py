#
# A basic neural network implementation written in numpy. Because I would prefer to understand how it works before using a library to do it.
# (also beause numpy arrays and torch tensors are largely compatible)
# Inspired by this excellent explanation of how a basic neural net works: https://iamtrask.github.io/2015/07/12/basic-python-network/
# Functionality expanded significantly.
#

import numpy as np

SHAPE = [16, 8, 4, 2]
ITERS = 100000

def sigmoid(x, deriv=False):
	if deriv:
		return x * (1-x)
	return 1 / (1 + np.exp(-x))

def relu(x, deriv=False):
	if deriv:
		return 1 * (x > 0)
	return x * (x > 0)

np.random.seed(1)
data = np.random.randint(10,size=(50, 10)) #A bunch of randomly generated data.

out = data.min(axis=1)/10 #Learning to take the minimum. Divided by 10 to get it between 0 and 1.
out = out.reshape(out.shape[0],-1) #Shape the data correctly.

np.random.seed(1)
SHAPE.append(out.shape[1])
synapses = list()
layers = list()
layers.append(data)
prev = data.shape[1]
for size in SHAPE:
	synapses.append(2*np.random.random((prev, size))-1)
	layers.append(np.zeros((data.shape[0], size)))
	prev = size

for i in range(ITERS):
	for l in range(len(SHAPE)):
		layers[l+1] = sigmoid(np.dot(layers[l], synapses[l]))
	if (i%10000) == 0:
		print("Error:" + str(np.mean(np.abs(out - layers[-1]))))
	prev_delta = (out - layers[-1]) * sigmoid(layers[-1], deriv=True)
	for l in range(len(SHAPE)-1, -1, -1):
		synapses[l] += layers[l].T.dot(prev_delta)
		prev_delta = (prev_delta.dot(synapses[l].T)) * sigmoid(layers[l], deriv=True)

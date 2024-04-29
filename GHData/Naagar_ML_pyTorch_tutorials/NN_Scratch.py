# NN form Scratch in python 
# print('hello NN')
import numpy as np 

def sigmoid_derivatives(x):
    return x * (1 - x)
def sigmoind(x):
    return 1 / (1 + np.exp(-x))
traning_input  = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]]) 
traning_output = np.array([[0, 1, 1, 0]]).T
np.random.seed(1)
synaptic_weights = 2 * np.random.random((3, 1)) - 1
print('Random starting synaptic weights:', synaptic_weights)

for i in range(2000):
    input_layers = traning_input
    output = sigmoind(np.dot(input_layers, synaptic_weights))
    error = traning_output - output
    adjustemt = error * sigmoid_derivatives(output) 
    synaptic_weights += np.dot(input_layers.T, adjustemt)
    # print("outputs: ",output)
# adjested weights by error weighted derivates


print("synaptic_weights after traning",synaptic_weights)


#Learning with numpy using tutorial from enlight.nyc
import numpy as np

#Toy dataset
X = np.array([[2,9],[1,5],[3,6]], dtype=float)
Y = np.array([[92], [86], [89]], dtype=float)

#Scale data
X = X/np.amax(X, axis=0)
Y = Y/100

#Define Neural Network class
class Neural_Network(object):
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)
    
    #Forward propagation
    def forward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o
    
    #Sigmoid
    def sigmoid(self, s):
        return 1/(1+np.exp(-s))
        
    #Derivative of sigmoid
    def sigmoidPrime(self, g):
        return g*(1-g)
        
    #Backward propagation
    def backward(self, X, Y, o):
        #Output error
        self.o_error = Y-o
        #Delta at output layer
        self.o_delta = self.o_error*self.sigmoidPrime(o)
        
        #Error at hidden layer
        self.z2_error = self.o_delta.dot(self.W2.T)
        #Delta at hidden layer
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)
        
        #Update weights
        self.W1 += 2*X.T.dot(self.z2_delta)
        self.W2 += 2*self.z2.T.dot(self.o_delta)
    
    #Train
    def train(self, X, Y):
        o = self.forward(X)
        self.backward(X,Y,o)
        
        
NN = Neural_Network()
for i in range(10000): # trains the NN 1,000 times
    print("Input: \n" + str(X))
    print("Actual Output: \n" + str(Y))
    print("Predicted Output: \n" + str(NN.forward(X)))
    print("Loss: \n" + str(np.mean(np.square(Y - NN.forward(X))))) # mean sum squared loss
    print("\n")
    NN.train(X, Y)
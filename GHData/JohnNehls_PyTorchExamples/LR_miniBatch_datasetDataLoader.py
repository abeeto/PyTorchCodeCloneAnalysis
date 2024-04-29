#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

#set the random seed for the script
np.random.seed(0)

#create noisy data
class noisyLineData(Dataset):
    def __init__(self, N=100, slope=2, intercept=3, stdDev=50):
        self.x = torch.linspace(-100,100,N) 
        self.y = slope*self.x + intercept + np.random.normal(0, stdDev, N) #can use numpy for random
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)        

data = noisyLineData()

trainloader = DataLoader(dataset = data, batch_size = 20)

#Global fit parameters w and b, and the prediction vector yhat
modelParams = {'w': torch.tensor(float(0),requires_grad = True),
              'b': torch.tensor(float(0),requires_grad = True)  }

#Learning parameters
lr = 1e-4 #learning Rate 
epochs = 4 # number of iterations

#general function which use and modify the global parameters
def forward(x, modelParams):
    """Forward step is the prediction step. Where the input data (x) is multiplied
    by the parameters (w) and added to the bias (b), resulting in th prediction, yhat."""
    
    return x * modelParams['w'] + modelParams['b']

def criterion(yhat, y):
    """Criterion is a measure of the error betwen the prediction (yhat) and the data (y).
    L2 error is likely the most common. Greater the power, the more weight to outliers. """
    
    return torch.mean( ( yhat - y )**2 )

def backward(loss, modelParams, lr):
    """The backward step is the the optimization step. We calculate the gradient of the
    loss w.r.t. the model parameters (w and b) and travel in the negative gradient direction
    to minimize the loss. Simple Gradient Descent. """
    
    # tells the tree to calculate the parial derivates of the loss wrt all of the
    #contriubuting tensors with the "requires_grad = True" in their constructor.
    loss.backward() 
    
    #gradient descent 
    modelParams['w'].data = modelParams['w'].data - lr * modelParams['w'].grad.data
    modelParams['b'].data = modelParams['b'].data - lr * modelParams['b'].grad.data
    
    #must zero out the gradient otherwise pytorch accumulates the gradient. 
    modelParams['w'].grad.data.zero_()
    modelParams['b'].grad.data.zero_()

##lists to save the parameters and errors
params = []
error = []
error_epoch = []

for epoch in range(epochs):
    yhat_total = forward(data.x, modelParams)
    error_epoch.append(float(criterion(yhat_total, data.y).data))

    params.append([modelParams['w'].data, modelParams['b'].data, epoch]) # saving data    
    # mini-batch or stochastic gradient descent
    for x,y in trainloader:        
        yhat = forward(x, modelParams) #major step 1/3
        
        loss = criterion(yhat, y) #major step 2/3

        error.append(loss.data) #saving data

        backward(loss, modelParams, lr) #major step 3/3    

#saving data
params.append( [modelParams['w'].data, modelParams['b'].data, epochs] )
error.append( criterion( yhat, y ).data )
params = np.array(params)
error = np.array(error)

#Simple display of the learning 
print(error)
print(error_epoch)

plt.figure()
plt.plot(data.x.numpy(), data.y.numpy(), 'xk', label="data")
for param in params:
    plt.plot(data.x.numpy(),param[0]*data.x.numpy()+param[1], label = f'epoch {int(param[2])}')
plt.legend()
plt.title("mini-batch gradient descent with PyTorch")
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./figs/LR_miniBatch_datasetDataLoader.png')

plt.figure()
plt.plot(error)
plt.title("mini-batch gradient descent with PyTorch")
plt.xlabel('batch')
plt.ylabel('Loss')

#importing the dependencies

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

x_values = [i for i in range(11)] #creating a small dataset for our program
x_train = np.array(x_values, dtype=np.float32)#converting it into numpy arrays
x_train = x_train.reshape(-1, 1)#reshaping it with a fake dimension because this is how torch Variable accepts a numpy array .

y_values = [2*i + 1 for i in x_values]#creating a small dataset for our program
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

'''
CREATE MODEL CLASS
'''
class LinearRegressionModel(nn.Module): # we import all the stuffs of nn.module which automaticaly get fused into our class
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()# importing the constructor/initiator of nn.module from superclass
        self.linear = nn.Linear(input_dim, output_dim)  # linear model 
    
    def forward(self, x): # just a common forward method 
        out = self.linear(x)
        return out

'''
INSTANTIATE MODEL CLASS
'''
input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim, output_dim)


#######################
#  USE GPU FOR MODEL  #
#######################

model.cuda() # if we want to use gpu for acceleration

'''
INSTANTIATE LOSS CLASS
'''

criterion = nn.MSELoss() # we use the mean square error loss 

'''
INSTANTIATE OPTIMIZER CLASS
'''

learning_rate = 0.01 # we should use a small learning rate so that , the model doesn't skip any better weights

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # we use the Stochastic Gradient Descent as our optimizer

'''
TRAIN THE MODEL
'''
epochs = 100
for epoch in range(epochs):
    epoch += 1
    # Convert numpy array to torch Variable
    
    #######################
    #  USE GPU FOR MODEL  #
    #######################
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(x_train).cuda())
        
    #######################
    #  USE GPU FOR MODEL  #
    #######################
    if torch.cuda.is_available():
        labels = Variable(torch.from_numpy(y_train).cuda())
        
    # Clear gradients w.r.t. parameters
    optimizer.zero_grad() 
    
    # Forward to get output
    outputs = model(inputs)
    
    # Calculate Loss
    loss = criterion(outputs, labels)
    
    # Getting gradients w.r.t. parameters
    loss.backward()
    
    # Updating parameters
    optimizer.step()
    
    # Logging
    print('epoch {}, loss {}'.format(epoch, loss.data[0]))

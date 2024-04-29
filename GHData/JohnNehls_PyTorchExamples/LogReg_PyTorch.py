#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

#create noisy data
class NoisyBinaryData(Dataset):
    def __init__(self, N=100, x0=-3, x1=5, stdDev=2):                
        xlist = []; ylist = []
        for i in range(N):
            #class 0
            if np.random.rand()<0.5:
                xlist.append(np.random.normal(x0,stdDev))
                ylist.append(0.0)
            #class 1
            else:
                xlist.append(np.random.normal(x1,stdDev))
                ylist.append(1.0)

        self.x = torch.tensor(xlist).view(-1,1)
        self.y = torch.tensor(ylist).view(-1,1)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)        

np.random.seed(0)
data = NoisyBinaryData()
trainloader = DataLoader(dataset = data, batch_size = 20)

# create my "own" linear regression model 
class logistic_regression(nn.Module):
    def __init__(self, input_size, output_size):
        #call the super's constructor and use it without having to store it directly. 
        super(logistic_regression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        """Prediction"""
        return torch.sigmoid(self.linear(x))
    
# ERROR with NANS
# def criterion(yhat,y):
#      out = -1 * torch.mean(y * torch.log(yhat) + (1 - y) * torch.log(1 - yhat))
#      return out
criterion = nn.BCELoss()

model = logistic_regression(1,1)
model.state_dict()['linear.weight'][0] = 0.0
model.state_dict()['linear.bias'][0] = 0.5 

optimizer = optim.SGD(model.parameters(), lr = 2)

def train_model(epochs):
    ERROR = []
    PARAMS = []

    for epoch in range(epochs+1):
                
        PARAMS.append([model.state_dict()['linear.weight'].numpy()[0][0],
                           model.state_dict()['linear.bias'].numpy()[0], epoch])

        for x,y in trainloader:            
            yhat = model(x)
            loss = criterion(yhat,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ERROR.append(loss.data)
    
    return PARAMS, ERROR


PARAMS, ERROR = train_model(1000)

#Simple display of the learning 
plt.figure()
plt.plot(data.x.numpy(), data.y.numpy(), 'xk', label="data")
for param in PARAMS:
    if param[2] in [0, 1, 10, 50, 1000]:
        plt.plot(data.x.numpy(),param[0]*data.x.numpy()+param[1], label = f'epoch {int(param[2])}')
plt.legend(loc='upper left')
plt.title("Logistic Regression with PyTorch")
plt.xlabel('x')
plt.ylabel('y')
plt.ylim([-0.5,1.5])
plt.savefig('./figs/LogReg_PyTorch.png',dpi=300)


plt.figure()
plt.plot(ERROR)
plt.title("Error")
plt.xlabel('batch')
plt.ylabel('error')

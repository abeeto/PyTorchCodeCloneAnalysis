#!/usr/bin/env python

import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

#set the random seed for the script
torch.manual_seed(0)

#create noisy data
class noisyLineData(Dataset):
    def __init__(self, N=100, slope=2, intercept=3, stdDev=50):
        self.x = torch.linspace(-100,100,N).view(-1,1)
        noise = torch.normal( mean=torch.zeros(N), std= stdDev * torch.ones(N) ).view(-1,1)
        self.y = slope*self.x + intercept + noise
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)        

data = noisyLineData()
trainloader = DataLoader(dataset = data, batch_size = 15)

# create my "own" linear regression model 
class linear_regression(nn.Module):
    def __init__(self, input_size, output_size):
        #call the super's constructor and use it without having to store it directly. 
        super(linear_regression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        """Prediction"""
        return self.linear(x)
    
criterion = nn.MSELoss()

model = linear_regression(1,1)
model.state_dict()['linear.weight'][0] = 0
model.state_dict()['linear.bias'][0] = 0 

optimizer = optim.SGD(model.parameters(), lr = 1e-4)

def train_model(epochs):

    ERROR = []
    PARAMS = []

    for epoch in range(epochs):

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


PARAMS, ERROR = train_model(4)

#Simple display of the learning 
plt.figure()
plt.plot(data.x.numpy(), data.y.numpy(), 'xk', label="data")
for param in PARAMS:
    plt.plot(data.x.numpy(),param[0]*data.x.numpy()+param[1], label = f'epoch {int(param[2])}')
plt.legend()
plt.title("mini-batch gradient descent with PyTorch")
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./figs/LR_miniBatch_PyTorchway.png')

plt.figure()
plt.plot(ERROR)
plt.title("mini-batch gradient descent with PyTorch")
plt.xlabel('batch')
plt.ylabel('loss')

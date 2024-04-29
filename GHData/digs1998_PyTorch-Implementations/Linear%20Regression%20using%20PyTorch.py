#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


# In[12]:


'''
create datasets, for model predictions
forward pass
create optimizer and loss
training loop
- include loss and optimizer
- calculate gradients using backpropagation
- update weights
- repeat
'''
# creating datasets
x_numpy, y_numpy = datasets.make_regression(n_samples=100, 
                                            n_features=1,
                                           noise=20, 
                                            random_state=45)
#converting to torch tensors
x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1) #to reshape into 1 column

#samples and features
n_samples, n_features = x.shape

#model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

#creating loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

#training loop
n_iters = 150
for epoch in range(n_iters):
    #forward prop
    y_pred = model(x)
    loss = criterion(y_pred, y)
    
    #backward
    loss.backward()
    
    #update the weights
    optimizer.step()
    
    #empty gradients to prevent summing up
    optimizer.zero_grad()
    
    if (epoch+1) % 10 == 0:
        print(f'epoch : {epoch+1}, loss = {loss.item():.4f}')
        
#plot after converting torch tensors to numpy
prediction = model(x).detach().numpy() #to prevent gradient computation in plot
plt.plot(x_numpy, y_numpy, 'ro')
plt.plot(x_numpy, prediction, 'b')
plt.show()


# In[ ]:





# Creating a model that predicts crop yields for apples and oranges (target variables)
# by looking at the average temperature, rainfall, and humidity (input variables or features)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70], 
                   [74, 66, 43], 
                   [91, 87, 65], 
                   [88, 134, 59], 
                   [101, 44, 37], 
                   [68, 96, 71], 
                   [73, 66, 44], 
                   [92, 87, 64], 
                   [87, 135, 57], 
                   [103, 43, 36], 
                   [68, 97, 70]], dtype='float32')

targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119],
                    [57, 69], 
                    [80, 102], 
                    [118, 132], 
                    [21, 38], 
                    [104, 118], 
                    [57, 69], 
                    [82, 100], 
                    [118, 134], 
                    [20, 38], 
                    [102, 120]], dtype='float32')

# Numpy to Torch tensor conversion
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# Dataset
train_ds = TensorDataset(inputs, targets)

# DataLoader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

#Linear Regression Model
input_size = 3
output_size = 2
model = nn.Linear(input_size, output_size)

# Loss and Optimiser
loss_fn = F.mse_loss
opt = torch.optim.SGD(model.parameters(), lr=1e-5)

# Train the model
epochs = 1000
total_step = len(train_dl)
for epoch in range(epochs):
    for i, (inputs, targets) in enumerate(train_dl):

        ## Forward Pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        ## Backward and Optimize
        opt.zero_grad()
        loss.backward()
        opt.step()
    
        if (epoch+1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
# Input (temp, rainfall, humidity)
inputs = torch.tensor([[73.0, 67, 43], 
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
                   [68, 97, 70]])

# Targets (apples, oranges)
targets = torch.tensor([[56.0, 70], 
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
                    [102, 120]])
train_ds = TensorDataset(inputs,targets)
batch_size=5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
model = nn.Linear(3,2)
print(model.weight)
print(model.bias)
preds= model(inputs)

loss_fn = F.mse_loss
loss = loss_fn(model(inputs) ,targets)

opt = torch.optim.SGD(model.parameters(), lr=1e-5)

def fit(num_epochs, model, loss_fn, opt, train_dl):
    for epoch in range(num_epochs):
        for x,y in train_dl:
            preds=model(x)
            loss=loss_fn(preds, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
            
        



fit(800, model, loss_fn, opt, train_dl)

preds=model(inputs)
preds

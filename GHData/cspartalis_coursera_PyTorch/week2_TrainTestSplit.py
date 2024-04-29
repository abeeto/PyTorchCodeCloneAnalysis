import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class Data(Dataset):
    def __init__(self,train=True):
        self.x=torch.arange(-3,3,0.1).view(-1,1)
        self.f=-3*self.x+1
        self.y=self.f+0.1*torch.randn(self.x.size())
        self.len=self.x.shape[0]
        if train==True:     # Create outliers for train set.
            self.y[0]=0
            self.y[50:55]=20
        else:
            pass

    def __getitem__(self,index):
        return self.x[index],self.y[index]

    def __len__(self):
        return self.len

class LR(nn.Module):
    def __init__(self,input_size,output_size):
        super(LR,self).__init__()
        self.linear=nn.Linear(input_size,output_size)

    def forward(self,x):
        out=self.linear(x)
        return out

train_data=Data()
val_data=Data(train=False)

epochs=10
learning_rates=[0.0001,0.001,0.01,0.1,1]

criterion=nn.MSELoss()
trainloader=DataLoader(dataset=train_data,batch_size=1)

validation_error=torch.zeros(len(learning_rates))
test_error=torch.zeros(len(learning_rates))

MODELS=[]
for i,learning_rate in enumerate(learning_rates):
    model=LR(1,1)
    optimizer=optim.SGD(model.parameters(),lr=learning_rate)

    for epoch in range(epochs):
        for x,y in trainloader:
            yhat=model(x)
            loss=criterion(yhat,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    yhat=model(train_data.x)
    loss=criterion(yhat,train_data.y)
    test_error[i]=loss.item()
    MODELS.append(model)
    yhat=model(val_data.x)
    loss=criterion(yhat,val_data.y)
    validation_error[i]=loss.item()
    MODELS.append(model)
   
# Plot Cost/Total Loss
plt.semilogx(np.array(learning_rates),validation_error.numpy(),
        label='training cost/total loss')
plt.semilogx(np.array(learning_rates),test_error.numpy(),
        label='validation cost/total loss')
plt.ylabel('Cost/Total Loss')
plt.xlabel('leaning rate')
plt.legend()
plt.show()

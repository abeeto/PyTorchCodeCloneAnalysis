import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import Dataset,DataLoader

class Data(Dataset):
    def __init__(self):
        self.x=torch.arange(-3,3,0.1).view(-1,1)
        self.y=3*X+1
        self.len=self.x.shape[0]

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

criterion=nn.MSELoss()
w=torch.tensor(-15.0,requires_grad=True)
b=torch.tensor(-10.0,requires_grad=True)
X=torch.arange(-3,3,0.1).view(-1,1)

dataset=Data()
trainloader=DataLoader(dataset=dataset,batch_size=1)
model=LR(1,1)
optimizer=optim.SGD(model.parameters(),lr=0.01)
print(optimizer.state_dict())

for epoch in range(100):
    for x,y in trainloader:
        yhat=model(x)
        loss=criterion(yhat,y)
        optimizer.zero_grad()   # always begin w/ this
        loss.backward()
        optimizer.step()        # update parameters


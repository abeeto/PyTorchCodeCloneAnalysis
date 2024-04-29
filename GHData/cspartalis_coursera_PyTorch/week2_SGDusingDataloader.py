import torch
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt

class Data(Dataset):
    def __init__(self):
        self.x=torch.arange(-3,3,0.1).view(-1,1)
        self.y=3*X+1
        self.len=self.x.shape[0]

    def __getitem__(self,index):
        return self.x[index],self.y[index]

    def __len__(self):
        return self.len

def forward(x):
    y=w*x+b
    return y

def criterion(yhat,y):
    return torch.mean((yhat-y)**2)

w=torch.tensor(-15.0,requires_grad=True)
b=torch.tensor(-10.0,requires_grad=True)
X=torch.arange(-3,3,0.1).view(-1,1)

dataset=Data()
trainloader=DataLoader(dataset=dataset,batch_size=5)

lr=0.1
COST=[]
for x,y in trainloader:
    yhat=forward(x)
    loss=criterion(yhat,y)
    loss.backward()
    w.data=w.data-lr*w.grad.data
    b.data=b.data-lr*b.grad.data
    w.grad.data.zero_()
    b.grad.data.zero_()
    COST.append(loss.detach().numpy())

plt.plot([i for i in range(len(COST))],COST)
plt.show()
 
 

 
 
 
 
 
 
 
 

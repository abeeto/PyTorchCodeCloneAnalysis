import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class dataset(Dataset):
    def __init__(self):
        self.x=torch.arange(-1,1,0.1)
        self.y=torch.arange(5.6,7.6,0.1)
    def __getitem__(self, index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.x.shape[0]

class logistig_regression(nn.Module):
    def __init__(self,in_size,out_size=1):
        super(logistig_regression,self).__init__()
        self.linear = nn.Linear(in_size,out_size)

    def forward(self,x):
        x = torch.sigmoid(self.linear(x))
        return x

dataset=dataset()
trainloader = DataLoader(dataset=dataset, batch_size=1)
model=logistig_regression(in_size=1)
criterion=nn.BCELoss()
optimizer=optim.SGD(model.parameters(),lr=0.01)
for epoch in range(100):
    for x,y in trainloader:
        yhat = model(x)
        loss = criterion(yhat,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self,D_in,H,D_out):
        super(Net,self).__init__()
        self.linear1 = nn.Linear(D_in,H)
        self.linear2 = nn.Linear(H,D_out)

    def forward(self,x):
        x=nn.Sigmoid(self.linear1(x))
        x=nn.Sigmoid(self.linear2(x))
        return x

def train(Y,X,model,optimizer,criterion,epochs=1000):
    cost=[]
    for epoch in range(epochs):
        total=0
        for y,x in zip(Y,X):
            yhat=model(x)
            loss=criterion(yhat,y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #cumulative loss
            total+=loss.item()
        cost.append(total)
    return cost

criterion = nn.BCELoss()
X = torch.arange(-20,20,1).view(-1,1).type(torch.FloatTensor)
Y = torch.zeros(X.shape[0])
Y[(X[:,0]>-4) & (X[:,0]<4)]=1.0

model = Net(1,2,1)
optimizer = optim.SGD(model.parameters(),lr=0.01)
cost = train(Y,X,model,optimizer,criterion,epochs=100)
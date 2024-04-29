import torch
import matplotlib.pyplot as plt

def forward(x):
    y=w*x+b
    return y

def criterion(yhat,y):
    return torch.mean((yhat-y)**2)

w=torch.tensor(-15.0,requires_grad=True)
b=torch.tensor(-10.0,requires_grad=True)
X=torch.arange(-3,3,0.1).view(-1,1)
f=-3*X
Y=f+0.1*torch.randn(X.size())

lr=0.1
COST=[]
for epoch in range(4):
    total=0
    for x,y in zip(X,Y):
        yhat=forward(x)
        loss=criterion(yhat,y)
        loss.backward()
        w.data=w.data-lr*w.grad.data
        b.data=b.data-lr*b.grad.data
        w.grad.data.zero_()
        b.grad.data.zero_()
        total+=loss.item()
    COST.append(total)

plt.plot([i for i in range(len(COST))],COST)
plt.show()

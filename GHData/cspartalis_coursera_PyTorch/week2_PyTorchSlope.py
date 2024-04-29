import torch
import matplotlib.pyplot as plt

def forward(x):
    return w*x

def criterion(yhat,y):
    return torch.mean((yhat-y)**2)

w=torch.tensor(-10.0,requires_grad=True)
X=torch.arange(-3,3,0.1).view(-1,1)     # x dimension
f=-3*X                                  
Y=f+0.1*torch.randn(X.size())           # y dimension

lr=0.1
COST=[]
for epoch in range(4):
    Yhat=forward(X)
    loss=criterion(Yhat,Y)  # It's the Cost function (:average of Losses)
    loss.backward()         # Calculates derivative w/ respect to all variables
    w.data=w.data-lr*w.grad.data
    w.grad.data.zero_()     # Setting grad=0 for the next iteration
    COST.append(loss.item())

plt.plot([0,1,2,3],COST)
plt.show()

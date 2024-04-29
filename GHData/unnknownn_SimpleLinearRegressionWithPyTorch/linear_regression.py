import torch
import matplotlib.pyplot as plt
import numpy as np

x = torch.arange(-3,3,0.1).view(-1,1) 

f = -3*x

plt.plot(x,f,label='f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#adding noise to f(x)

y = f + 0.1 * torch.randn(x.size())

#plot data points
plt.plot(x,y,'rx',label='f')

plt.plot(x,f,label='f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

class LRModel:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.w = torch.tensor(-10.0,requires_grad=True)
        self.b = torch.tensor(-15.0,requires_grad=True)
        self.lr = 0.1
        self.LOSS = []

    def forward(self,x):
        return x * self.w + self.b
    
    def criterion(self,yhat,y): #mse func to evalute the result
        return torch.mean((yhat-y)**2)
    
    def train(self,iter):
        for epoch in range(iter):
            yhat = self.forward(self.x)

            loss = self.criterion(yhat,self.y)
            
            self.LOSS.append(loss)

            loss.backward()

            #update parameters
            self.w.data = self.w.data - self.lr*self.w.grad.data

            self.b.data = self.b.data - self.lr*self.b.grad.data

            #zero the gradients before running the backward pass
            self.w.grad.data.zero_()
            self.b.grad.data.zero_()

    def visualize_loss_function(self):
        plt.plot(self.LOSS)
        plt.tight_layout()
        plt.xlabel("Epoch/Iterations")
        plt.ylabel("Cost")
        plt.show()

LR = LRModel(x,y)

LR.train(4)

LR.visualize_loss_function()


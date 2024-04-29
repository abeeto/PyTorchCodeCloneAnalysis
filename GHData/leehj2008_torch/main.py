import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.tensor import Tensor
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
#Give Data
X=np.random.rand(1000)
Y=np.random.rand(1000)
Z=X**2+Y**3+10
#Data End

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        self.fc1=nn.Linear(1,1000)
        self.fc2=nn.Linear(500,20)
        self.out=nn.Linear(1000,1)
        
    def forward(self, x):
        x=F.relu(self.fc1(x))
        #x=F.relu(self.fc2(x))       
        x=self.out(x)
        return x
    
net = Net()
print net
        
from torch import optim
criterion = nn.MSELoss()
optimizer=optim.SGD(net.parameters(),lr=0.0001)
for epoch in xrange(10000):
    optimizer.zero_grad()
    X=(np.random.rand(1000,1)-0.5)*10
    #if epoch%2==0:
    #X=-X
    X=Variable(Tensor(X))
    #print X
    Z=Variable(Tensor(np.sin(X)))
    #print Z
    out=net(Variable(Tensor(X)))
    loss = criterion(out,Z)
    print epoch,loss
    loss.backward()
    optimizer.step();

X=np.linspace(-5, 5, 200).reshape(200,1)
tX=Tensor(X)
z=np.sin(X)
print net(tX),z


plt.figure(0)
plt.plot(X,z)
plt.plot(X,net(tX).data.numpy(),color='r')
plt.show()
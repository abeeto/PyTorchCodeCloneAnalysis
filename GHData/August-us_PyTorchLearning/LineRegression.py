import torch
from torch.autograd import Variable
from torch.nn import functional as F
from matplotlib import pyplot as plt


x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y=x.pow(2)+0.2*torch.rand(x.size())
print (x)

x.y=Variable(x),Variable(y)

class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_output=1):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(n_features,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x

net=Net(1,10,1)
plt.ion()  # print on time
plt.show()
optimizer=torch.optim.SGD(net.parameters(),lr=0.5)
loss_func=torch.nn.MSELoss()   # the regression often ueses
for t in range(100):
    prediction=net(x)
    loss=loss_func(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t%5==0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=0.5)
        plt.text(0.5,0,'LOSS=%.4f'%loss.item(),fontdict={'size':20,'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
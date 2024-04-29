import torch
from torch.autograd import Variable
from torch.nn import functional as F
from matplotlib import pyplot as plt


n_data=torch.ones(100,2)
x0=torch.normal(2*n_data,1)# normal(means,std)
y0=torch.zeros(100)
x1=torch.normal(-2*n_data,1)
y1=torch.ones(100)
x=torch.cat((x0,x1),0).type(torch.FloatTensor)
y=torch.cat((y0,y1),).type(torch.LongTensor)


x.y=Variable(x),Variable(y)
# plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=y.data.numpy(),s=100,lw=0,cmap=)
# plt.show()

class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_output=1):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(n_features,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x

net=Net(2,10,2)# input is 2d, output is 2d
plt.ion()  # print on time
plt.show()
optimizer=torch.optim.SGD(net.parameters(),lr=0.02)
loss_func=torch.nn.CrossEntropyLoss()
for t in range(100):
    out=net(x)
    loss=loss_func(out,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t%2==0:
        plt.cla()
        prediction=torch.max(F.softmax(out),1)[1] # index=0 is value ,index =1 is the position
        pred_y=prediction.data.numpy().squeeze()
        traget_y=y.data.numpy()
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=pred_y,s=100,lw=0)
        accuracy=sum(pred_y==traget_y)/200.0
        plt.text(1.5,-4,'Accuracy=%0.2f'%accuracy,fontdict={'size':20,'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
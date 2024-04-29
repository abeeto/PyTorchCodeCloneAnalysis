import torch
import torch.tensor as tensor
import numpy as np
rng=np.random.RandomState(23455)
X=rng.rand(32,2)
X=X.astype(np.float32)
Y=np.array([[int(x0+x1<1)] for [x0,x1] in X],dtype=np.float32)
X=tensor(X)
Y=tensor(Y)

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden1,n_output):
        super(Net,self).__init__()
        self.hidden1=torch.nn.Linear(n_feature,n_hidden1)
        self.output=torch.nn.Linear(n_hidden1,n_output)
        self.sigmoid=torch.nn.Sigmoid()
    def forward(self, x):
        h1=self.hidden1(x)
        output=self.output(h1)
        y=self.sigmoid(output)
        return y

net=Net(2,3,1)
optimizer=torch.optim.SGD(net.parameters(),lr=0.01)
loss_func=torch.nn.MSELoss()

for t in range(3000):
    prediction=net(X)
    loss=loss_func(prediction,Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t%50==0:
        print(loss.data.numpy())
test=np.array([[1.0,2.0]],dtype=np.float32)
pre=net(tensor(test))
print(pre.data.numpy())
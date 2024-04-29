import torch
import torch.utils.data as Data
import torch.tensor as tensor
import numpy as np
rng=np.random.RandomState(23455)
X=rng.rand(32,2)
X=X.astype(np.float32)
Y=np.array([[int(x0+x1<1)] for [x0,x1] in X],dtype=np.float32)
X,Y=tensor(X),tensor(Y)

torch_dataset=Data.TensorDataset(X,Y)
loader=Data.DataLoader(
    dataset=torch_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)



#自定义损失函数
class My_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.pow((x - y), 2))




class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_output):
        super(Net,self).__init__()
        self.h1=torch.nn.Linear(n_features,n_hidden)
        self.output=torch.nn.Linear(n_hidden,n_output)
        self.sigmoid=torch.nn.Sigmoid()
    def forward(self, x):
        h1=self.h1(x)
        output=self.output(h1)
        y=self.sigmoid(output)
        return y

net=Net(2,3,1)
optimizer=torch.optim.SGD(net.parameters(),lr=0.01)
#loss_func=torch.nn.MSELoss()
loss_func=My_loss()

for t in range(5000):
    for set,(batch_x,batch_y) in enumerate(loader):
        pred=net(batch_x)
        loss=loss_func(pred,batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    pred = net(X)
    loss = loss_func(pred, Y)
    print(loss.data.numpy())


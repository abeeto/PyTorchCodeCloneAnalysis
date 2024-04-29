import torch
import torch.nn as nn

N,D_in,H,D_out = 64,1000,100,10  #N表示样本的个数 ,D_in 表示样本的维数、H表示隐藏层的神经元个数、D_out表示样本的输出维度

x = torch.randn(N,D_in)  #制作一些随机数据 一共有64个 1000维的数据
y = torch.randn(N,D_out) #制作一些label


class TwoLayerNet(nn.Module):   #必须要继承nn.Module 这是固定写法
    def __init__(self,D_in,H,D_out):
        super(TwoLayerNet,self).__init__()   #固定写法
        self.linear1 = torch.nn.Linear(D_in,H,bias=False)
        self.linear2 = torch.nn.Linear(H,D_out,bias=False)
    def forward(self,x):
        y_pred = self.linear2(self.linear1(x).clamp(min=0))
        return y_pred

model = TwoLayerNet(D_in,H,D_out)   #初始化一个神经网络
loss_fn = nn.MSELoss(reduction="sum")  #
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

for it in range(500):
    y_pred = model(x)

    loss = loss_fn(y_pred,y)
    print(it,loss.item())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


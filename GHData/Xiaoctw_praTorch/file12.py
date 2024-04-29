import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.optim as optim
import random
"""
动态网络，每次运行网络都不一样，层数不同
"""
class DynamicNet(nn.Module):
    def __init__(self,D_in,H,D_out):
        #构建网络
        super(DynamicNet, self).__init__()
        self.input_linear=nn.Linear(D_in,H)
        #根据随机数的不同，可能传入多次中间层
        self.middle_linear=nn.Linear(H,H)
        self.output_linear=nn.Linear(H,D_out)

    def forward(self,x):
        """
        注意每次前向传播都会构建一个动态图，
        后向传播仍然依据这个图
        :param x:
        :return:
        """
        x=fun.relu(self.input_linear(x))
        for _ in range(random.randint(0,3)):
            x=fun.relu(self.middle_linear(x))
        x=self.output_linear(x)
        return x

N,D_in,H,D_out=64,1000,100,10
x=torch.randn(N,D_in)
y=torch.randn(N,D_out)

model=DynamicNet(D_in,H,D_out)

criterion=nn.MSELoss(reduction='sum')
optimizer=optim.SGD(model.parameters(),lr=1e-4,momentum=0.9)

for t in range(500):
    y_pred=model(x)
    loss=criterion(y_pred,y)
    print(t,loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



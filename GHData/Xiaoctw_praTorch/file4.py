import torch
import time
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
N,D_in,H,D_out=64,500,100,10
learning_rate=1e-5
x=Variable(torch.randn(N,D_in))
y=Variable(torch.randn(N,D_out),requires_grad=True)
x1=Variable(torch.randn(N,D_in))
y1=Variable(torch.randn(N,D_out),requires_grad=True)
model=nn.Sequential(
    nn.Linear(D_in,H),
    nn.ReLU(),
    nn.Linear(H,D_out)
)
loss_fn=nn.MSELoss(reduction='sum')
#采用了优化策略，利用Adam算法进行优化
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
#optimizer1=torch.optim.SGD(model.parameters(),lr=learning_rate)好像只找到了这两种优化策略
time1=time.time()
losses=[]
for t in range(1000):
    y_pred=model(x)
    loss=loss_fn(y_pred,y)
    optimizer.zero_grad()
    loss.backward()
    losses.append(loss)
    optimizer.step()  # 在这里就是对model的参数进行减去偏差
    print("{}次迭代，loss值为：{}".format(t,loss))
time2=time.time()
print("耗费时间{}".format(time2-time1))
plt.plot(losses,color='r')
plt.show()
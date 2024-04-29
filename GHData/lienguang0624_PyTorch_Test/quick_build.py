import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F

n_data = torch.ones(100,2)
x0 = torch.normal(2 * n_data,1)
y0 = torch.zeros(100)
x1 = torch.normal(-2 * n_data,1)
y1 = torch.ones(100)
x = torch.cat((x0,x1),0).type(torch.FloatTensor)
y = torch.cat((y0,y1),).type(torch.LongTensor)

x,y = Variable(x),Variable(y)

net = torch.nn.Sequential(
    torch.nn.Linear(2,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,2)
)
print(net)


# 定义优化方式（随机梯度下降）
optimizer = torch.optim.SGD(net.parameters(),lr = 0.001)

# 定义损失函数计算方式
loss_func = torch.nn.CrossEntropyLoss()

# 可视化
plt.figure()
plt.ion()
plt.show()

# 训练
for i in range(1000):
    out = net(x)
    loss = loss_func(out,y)
    optimizer.zero_grad()# 梯度设为零
    loss.backward()# 反向传递
    optimizer.step()# 优化梯度
    if i % 5 == 0:
        print('\nepoc:',i)
        plt.cla()
        prediction = torch.max(F.softmax(out),1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c = pred_y,s = 100,lw =0)
        plt.pause(0.1)

plt.ioff()
plt.show()



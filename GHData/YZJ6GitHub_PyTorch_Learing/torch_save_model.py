import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

def save():
    net = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    optimezer = torch.optim.SGD(net.parameters(),lr= 0.25)
    loss_func = torch.nn.MSELoss()
    for i in range(100):
        prediction = net(x)
        loss = loss_func(prediction,y)
        optimezer.zero_grad()
        loss.backward()
        optimezer.step()
    plt.figure(1,figsize=(10,3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    # 保存网路图结构以及参数， 保存整个网络
    torch.save(net,'net.pkl')
    # 只保存网络中的参数 (速度快, 占内存少)
    torch.save(net.state_dict(),'net_params.pkl')

#保存了整个网络，只需要加载模型就行
def restore_net():
    net2 = torch.load('net.pkl')
    prediction = net2(x)
    plt.figure(1, figsize=(10, 3))
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()

# save net1
save()

# restore entire net (may slow)
restore_net()

# restore only the net parameters
restore_params()


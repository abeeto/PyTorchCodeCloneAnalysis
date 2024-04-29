import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 超参数的设置
lr = 0.01
BATCH_SIZE = 20
EPOCH = 200

# optimizer = torch.optim.SGD()
a = torch.linspace(-1, 1, 1000)
# 升维操作
x = torch.unsqueeze(a, dim=1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))
# 绘制图像点
plt.scatter(x, y)
plt.show()
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)


# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(1, 20)
        self.predict = nn.Linear(20, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.predict(x)
        return x


# 定义优化器，实例网络
netSGD = Net()
optim_SGD = torch.optim.SGD(netSGD.parameters(), lr=lr)

netMomentum = Net()
optim_Momentum = torch.optim.SGD(netMomentum.parameters(), lr=lr, momentum=0.8)

netASGD = Net()
optim_ASGD = torch.optim.ASGD(netASGD.parameters(), lr=lr, )

netAdagrad = Net()
optim_Adagrad = torch.optim.Adagrad(netAdagrad.parameters(), lr=lr, )

netAdadelta = Net()
optim_Adadelta = torch.optim.Adadelta(netAdadelta.parameters(), lr=lr, )

netAdam = Net()
optim_Adam = torch.optim.Adam(netAdam.parameters(), lr=lr, betas=(0.9, 0.99))

netAdamW = Net()
optim_AdamW = torch.optim.AdamW(netAdamW.parameters(), lr=lr, )

# netLBFGS = Net()
# optim_LBFGS = torch.optim.LBFGS(netMomentum.parameters(),lr=lr,)

netRMSprop = Net()
optim_RMSprop = torch.optim.RMSprop(netRMSprop.parameters(), lr=lr, alpha=0.9)

netRprop = Net()
optim_Rprop = torch.optim.Rprop(netRprop.parameters(), lr=lr, )

# netSparseAdam = Net()
# optim_SparseAdam = torch.optim.SparseAdam(netSparseAdam.parameters(),lr=lr,)

nets = [netSGD, netMomentum, netASGD, netAdagrad, netAdadelta, netAdam, netAdamW, netRMSprop, netRprop, ]

optims = [optim_SGD, optim_Momentum, optim_ASGD, optim_Adagrad, optim_Adadelta, optim_Adam, optim_AdamW, optim_RMSprop,
          optim_Rprop, ]
loss_func = torch.nn.MSELoss()

losses = [[], [], [], [], [], [], [], [], [], ]

for epoch in range(EPOCH):
    for net, opt, ls in zip(nets, optims, losses):
        output = net(y)
        loss = loss_func(output, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ls.append(loss.data)

labels = ["SGD", 'ASGD', 'Momentum', 'Adadelta', "Adagrad", "Adam", "AdamW", "Adamax", "RMSprop", "Rprop", ]

# print("losses{}".format(losses))
for i, l in enumerate(losses):
    plt.plot(l, label=labels[i])
plt.legend(loc='best')
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.ylim((0, 0.2))
plt.show()

import torch
import torch.utils.data as Data

import torch.nn.functional as F

from torch.autograd import Variable
import matplotlib.pyplot as plt

# hyper paramerter

LR = 0.01
BATCH_SIZE = 32

EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))

# show dataset

plt.scatter(x.numpy(), y.numpy())

plt.show()

torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)

        return x


# different nets

net_SGD = Net()
net_mom = Net()
net_RMS = Net()
net_Adam = Net()

nets = [net_SGD, net_mom, net_RMS, net_Adam]

opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_mom = torch.optim.SGD(net_mom.parameters(), lr=LR, momentum=0.8)
opt_RMS = torch.optim.RMSprop(net_RMS.parameters(), lr=LR, alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))

opts = [opt_SGD,opt_mom,opt_RMS,opt_Adam]

loss_func = torch.nn.MSELoss()

loss_hits = [[],[],[],[]]

for epoch in range(EPOCH):
    print(epoch)
    for step, (bx,by) in enumerate(loader):
        b_x = Variable(bx)
        b_y = Variable(by)

        for net, opt,l_his in zip(nets,opts,loss_hits):
            output = net(b_x)
            loss = loss_func(output,b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_his.append(loss.data[0])



labels = ['SGD','MOM','RMS','Adam']

for i, l_his in enumerate(loss_hits):
    plt.plot(l_his,label=labels[i])
plt.legend(loc = 'best')

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim(0,0.2)
plt.show()

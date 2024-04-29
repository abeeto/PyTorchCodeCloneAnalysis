import torch
import numpy as np 
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data 

#HP
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

#FAKE DATA
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))  #*x.size()means here ge

# plt.scatter(x.numpy(), y.numpy())
# plt.show()

#input the data to torch dataset
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,)

#net
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

if __name__ == '__main__': #main function enter point
    #create different optimizer
    net_SGD = Net()
    net_Momenton = Net()
    net_RMSprop = Net()
    net_Adam = Net()
    nets = [net_SGD, net_Momenton, net_RMSprop, net_Adam]

    #train the network
    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momenton = torch.optim.SGD(net_Momenton.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    opts = [opt_SGD, opt_Momenton, opt_RMSprop, opt_Adam]
    loss_his = [[], [], [], []]

    #loss function
    loss_func = torch.nn.MSELoss()

    #train
    for epoch in range(EPOCH):
        print("Epoch:", epoch)
        for step, (b_x, b_y) in enumerate(loader):
            for net, opt, l_his in zip(nets, opts, loss_his):
                prediction = net(b_x)
                # print(net)
                # print(b_x.size())
                # print(b_y.size())
                # print(prediction)
                loss = loss_func(prediction, b_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                l_his.append(loss.data.numpy()) #add loss data into the final y
    #plot
    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for i, loss in enumerate(loss_his):
        plt.plot(loss, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim(0, 0.2)
    plt.show()

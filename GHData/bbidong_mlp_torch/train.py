import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from data_class import Data
import os
from tensorboardX import SummaryWriter


def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()

def compute_l2(output,label):
    """
    :param output: (n,2)
    :param label: (n,2)
    :return:
    """
    dis=output-label
    error=np.sqrt(dis[:,0]**2+dis[:,1]**2)
    return sum(error)

def make_fc(dim_in, hidden_dim, use_gn=False):
    '''
        Caffe2 implementation uses XavierFill, which in fact
        corresponds to kaiming_uniform_ in PyTorch
    '''
    if use_gn:
        fc = nn.Linear(dim_in, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(fc.weight, a=1)
        return nn.Sequential(fc, group_norm(hidden_dim))
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc


# 定义网络===============================================================================================================
class MapNet(nn.Module):

    def __init__(self):
        super(MapNet, self).__init__()

        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)
        nn.init.normal_(self.fc3.weight, std=0.001)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 参数设置
writer = SummaryWriter('result')  # 保存训练过程
num_epochs = 800
batch_size=16
learning_rate = 0.0003

guard='left' # 训练哪个哨兵的模型
trainset = Data(guard, 'train')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = Data(guard,'test')
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

net = MapNet().cuda()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

print("Start Training...")
for epoch in range(num_epochs):
    net.train()
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs=inputs.cuda()
        labels=labels.cuda()
        # 首先要把梯度清零，不然PyTorch每次计算梯度会累加，不清零的话第二次算的梯度等于第一次加第二次的
        optimizer.zero_grad()
        # 计算前向传播的输出
        outputs = net(inputs)
        # 根据输出计算loss
        loss = smooth_l1_loss(outputs, labels,1,False)
        loss = loss / len(labels)

        writer.add_scalar('loss', loss.item(), epoch*len(trainloader)+i)
        loss.backward()
        # 用计算的梯度去做优化
        optimizer.step()

        if i % 20 == 0:
            print('[Epoch %d, Batch %5d] loss: %.3f' % (epoch, i, loss))

    # 训练完一个epoch, 验证下精度
    with torch.no_grad():
        net.eval()
        total_error=0
        for i, data in enumerate(testloader):
            inputs, labels = data
            inputs = inputs.cuda()
            labels=labels.cuda()

            optimizer.zero_grad()
            outputs = net(inputs)

            outputs=outputs.cpu().numpy()
            labels=labels.cpu().numpy()
            # 计算L2误差
            error=compute_l2(outputs,labels)
            total_error+=error

        mean_error=total_error/len(testset)
        writer.add_scalar('l2_error', mean_error, epoch)

        print('[Epoch %d] l2_error: %.3f' % (epoch, mean_error))


print("Done Training!")

# 保存训练好的模型
torch.save(net, './model/mlp_'+guard+'.pt')

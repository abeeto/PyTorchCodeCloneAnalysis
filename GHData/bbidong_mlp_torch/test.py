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


# 定义网络======================================================
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


guard='left' # 使用哪个哨兵的模型
batch_size = 16

testset = Data(guard,'test')
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

# 加载训练好的模型
net_model = torch.load('./pretrain_model/mlp_'+guard+'.pt').cuda()
net_model.eval()

out_list=[]
lable_list=[]
with torch.no_grad():
    for i, data in enumerate(testloader):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        # 计算前向传播的输出
        outputs = net_model(inputs)
        outputs = outputs.cpu().numpy()
        labels = labels.cpu().numpy()

        out_list.extend(outputs)  # 模型输出
        lable_list.extend(labels)  # label

# ---------------可视化-----------------------------------------------
plt.figure()
out_list=np.array(out_list)
lable_list=np.array(lable_list)

vis_axis=0  # 要分析哪个轴的误差
# 把数据按label的axis轴排序
out_list = out_list[lable_list[:, vis_axis].argsort(), vis_axis]
lable_list = lable_list[lable_list[:, vis_axis].argsort(), vis_axis]

plt.scatter(range(len(out_list)),out_list,s=3,c='g',label='out')
plt.scatter(range(len(out_list)),abs(out_list-lable_list),s=3,c='b',label='error')

plt.plot(range(len(out_list)),lable_list,'r-',lw=2,label='label')

plt.legend(loc='upper left')
plt.show()
error=abs(out_list-lable_list)
mean_error=sum(error)/len(out_list)
print('mean error: '+ str(mean_error))


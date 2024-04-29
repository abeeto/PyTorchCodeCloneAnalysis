
import torch
from torch import nn
from torch.nn import init
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

N_SAMPLES = 2000
BATCH_SIZE = 64
EPOCH = 12
LR = 0.03
N_HIDDEN = 8
ACTIVATION = torch.tanh
B_INIT = -0.2

x = np.linspace(-7, 10, N_SAMPLES)[:, np.newaxis]
noise = np.random.normal(0, 2, x.shape)
y = np.square(x) - 5 + noise

test_x = np.linspace(-7, 10, 200)[:, np.newaxis]
noise = np.random.normal(0, 2, test_x.shape)
test_y = np.square(test_x) - 5 + noise

train_x = torch.from_numpy(x).float()
train_y = torch.from_numpy(y).float()
test_x = torch.from_numpy(x).float()
test_y = torch.from_numpy(y).float()

train_dataset = Data.TensorDataset(train_x, train_y)
train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# plt.scatter(train_x, train_y, c='#FF9359', s=50, alpha=0.2, label='train')
# plt.legend(loc='best')
# plt.show()

class Net(nn.Module):

    def __init__(self, batch_normalization=False):
        super(Net, self).__init__()
        self.do_bn = batch_normalization
        self.fcs = []                                       # 没有做批标准化的神经网络层结构
        self.bns = []                                       # 所有批标准化的神经网络层结构
        self.bn_input = nn.BatchNorm1d(1, momentum=0.5)     # 把输入的特征值在进入全连接层之前进行批标准化（不是必要操作，只要保证在激活函数前进行批标准化即可）

        for i in range(N_HIDDEN):
            input_size = 1 if i == 0 else 10
            fc = nn.Linear(input_size, 10)
            setattr(self, 'fc%i' % i, fc)                   # 便利设置对象属性（神经网络）
            self._set_init(fc)
            self.fcs.append(fc)
            if self.do_bn:
                bn = nn.BatchNorm1d(10, momentum=0.5)       # 增加批标准化层
                setattr(self, 'bn%i' % i, bn)
                self.bns.append(bn)

        self.predict = nn.Linear(10, 1)                     # 输出层
        self._set_init(self.predict)                        # 初始化输出层的参数

    def _set_init(self, layer):
        '''
        用于给神经网络层设置一个初始化的参数，这里的作用是检测一个错误的初始化参数对做了批标准化的神经网络和没做批标准化的神经网络的影响。
        :param layer: 某一层神经网络
        :return:
        '''
        init.normal_(layer.weight, mean=0, std=0.1)         # 以0为均值，0.1为标准差的正太分布来初始化层的权重
        init.constant_(layer.bias, B_INIT)                  # 初始化层的斜率为常数B_INIT的值（-0.2）

    def forward(self, x):
        pre_activation = [x]
        if self.do_bn: x = self.bn_input(x)                 # 初始输入的x进行批标准化
        layer_input = [x]
        for i in range(N_HIDDEN):
            x = self.fcs[i](x)                              # x经过全连接层
            pre_activation.append(x)                        # 把只经过全连接层的数据添加到pre_activation
            if self.do_bn: x = self.bns[i](x)               # x经过批标准化
            x = ACTIVATION(x)                               # x经过激活函数
            layer_input.append(x)                           # 把经过全连接层、批标准化（可选）、激活函数后的数据添加到layer_input
        out = self.predict(x)
        '''
        out是输出层的预测数据，
        layer_input经过激活函数后的数据，self.do_bn = True时，进行输入数据x的批标准化和所有隐藏层的批标准化
        pre_activation只经过全连接层的数据，self.do_bn = True时，不进行输入数据x的批标准化，只进行所有隐藏层的批标准化
        '''
        return out, layer_input, pre_activation

nets = [Net(batch_normalization=False), Net(batch_normalization=True)]

# print(*nets)

optim_list = [torch.optim.Adam(net.parameters(), lr=LR) for net in nets]

loss_func = torch.nn.MSELoss()

def plot_histogram(l_in, l_in_bn, pre_ac, pre_ac_bn):
    '''
    :param l_in: 经过所有全连接层（不包含最后的输出层）与激活函数层后的输出值，数组类型，内部有9个Tensor元素，分别表示输入层批标准化后的值（shape=(2000， 1)）和8个隐藏层（shape=(2000， 10)）的输出值
    :param l_in_bn: 过所有全连接层（不包含最后的输出层）、批标准化层与激活函数层后的输出值，数组类型，内部有9个Tensor元素，分别表示输入层批标准化后的值（shape=(2000， 1)）和8个隐藏层（shape=(2000， 10)）的输出值
    :param pre_ac: 经过所有全连接层（不包含最后的输出层）后的输出值，数组类型，内部有9个Tensor元素，分别表示输入层批标准化后的值（shape=(2000， 1)）和8个隐藏层（shape=(2000， 10)）的输出值
    :param pre_ac_bn: 经过所有全连接层（不包含最后的输出层）与批标准化层后的输出值，数组类型，内部有9个Tensor元素，分别表示输入层批标准化后的值（shape=(2000， 1)）和8个隐藏层（shape=(2000， 10)）的输出值
    :return:
    '''
    for i, (ax_pa, ax_pa_bn, ax, ax_bn) in enumerate(zip(axs[0, :], axs[1, :], axs[2, :], axs[3, :])):
        [a.clear() for a in [ax_pa, ax_pa_bn, ax, ax_bn]]                                                   # 遍历清除子图画布
        if i == 0:
            p_range = (-7, 10)
            the_range = (-7, 10)
        else:
            p_range = (-4, 4)
            the_range = (-1, 1)
        ax_pa.set_title('L' + str(i))       # 设置子图标题
        ax_pa.hist(pre_ac[i].data.numpy().ravel(), bins=10, range=p_range, color='#FF9359', alpha=0.5)      # 绘制直方图，在p_range区间内划分10个区间
        ax_pa_bn.hist(pre_ac_bn[i].data.numpy().ravel(), bins=10, range=p_range, color='#74BCFF', alpha=0.5)
        ax.hist(l_in[i].data.numpy().ravel(), bins=10, range=the_range, color='#FF9359')
        ax_bn.hist(l_in_bn[i].data.numpy().ravel(), bins=10, range=the_range, color='#74BCFF')
        for a in [ax_pa, ax, ax_pa_bn, ax_bn]:                                                              # 遍历清除子图刻度
            a.set_xticks(())
            a.set_yticks(())
        ax_pa_bn.set_xticks(p_range)                                                                        # 设置刻度区间
        ax_bn.set_xticks(the_range)
        '''
        通过下面的绘图可以发现，神经网络中错误的初始化参数，对做了批标准化的神经网络来说，影响不大。而对于没做批标准化的神经网络则会影响很大。
        '''
        axs[0, 0].set_ylabel('PreAct')                                                                      # 只经过全连接层的每个层输出值
        axs[1, 0].set_ylabel('BN PreAct')                                                                   # 经过全连接层与批标准化层的每个层输出值
        axs[2, 0].set_ylabel('Act')                                                                         # 经过全连接层与激活函数层的每个层输出值
        axs[3, 0].set_ylabel('BN Act')                                                                      # 经过全连接层、批标准化层、激活函数层的每个层输出值
    plt.pause(0.01)

if __name__ == '__main__':
    f, axs = plt.subplots(4, N_HIDDEN + 1, figsize=(10, 5))
    plt.ion()
    # plt.show()

    loss_list = [[], []]

    for epoch in range(EPOCH):
        print('Epoch', epoch)
        layer_inputs = []
        pre_acts = []
        # 检验模型所有隐藏层的数据分布
        for net, l in zip(nets, loss_list):
            net.eval()                                          # 神经网络转为预测模式
            pred, layer_input, pre_act = net(test_x)
            l.append(loss_func(pred, test_y).data.item())
            layer_inputs.append(layer_input)
            pre_acts.append(pre_act)
            net.train()                                         # 神经网络转为训练模式
        plot_histogram(*layer_inputs, *pre_acts)

        # 模型训练
        for step, (b_x, b_y) in enumerate(train_loader):
            for net, opt in zip(nets, optim_list):
                pred = net(b_x)[0]
                loss = loss_func(pred, b_y)
                opt.zero_grad()
                loss.backward()
                opt.step()

    plt.ioff()

    # 绘制loss
    plt.figure(2)
    plt.plot(loss_list[0], c='#FF9359', lw=3, label='Origial')
    plt.plot(loss_list[1], c='#74BCFF', lw=3, label='Batch Normalization')
    plt.xlabel('step')
    plt.ylabel('test loss')
    plt.ylim(0, 2000)
    plt.legend(loc='best')

    # 绘制拟合回归线
    [net.eval() for net in nets]
    preds = [net(test_x)[0] for net in nets]
    plt.figure(3)
    plt.plot(test_x.data.numpy(), preds[0].data.numpy(), c='#FF9359', lw=4, label='Original')
    plt.plot(test_x.data.numpy(), preds[1].data.numpy(), c='#74BCFF', lw=4, label='Batch Normalization')
    plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='r', s=50, alpha=0.2, label='train')
    plt.legend(loc='best')
    plt.show()
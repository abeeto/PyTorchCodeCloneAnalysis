
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

EPOCH = 500
N_SAMPLES = 20
N_HIDDEN = 300
LR = 0.01

x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
y = x + 0.3 * torch.normal(torch.zeros(N_SAMPLES, 1),torch.ones(N_SAMPLES, 1))

test_x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
test_y = test_x + 0.3 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

# plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5, label='train')
# plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label='test')
# plt.legend(loc='best')
# plt.ylim(-2.5, 2.5)
# plt.show()

net_normal = nn.Sequential(
    nn.Linear(1, N_HIDDEN),
    nn.ReLU(),
    nn.Linear(N_HIDDEN, N_HIDDEN),
    nn.ReLU(),
    nn.Linear(N_HIDDEN, 1)
)

net_drop = nn.Sequential(
    nn.Linear(1, N_HIDDEN),
    nn.Dropout(0.5),                     # 忽略掉50% 的神经元
    nn.ReLU(),
    nn.Linear(N_HIDDEN, N_HIDDEN),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(N_HIDDEN, 1)
)

print(net_normal)
print(net_drop)

optimizer_normal = torch.optim.Adam(net_normal.parameters(), lr=LR)
optimizer_drop = torch.optim.Adam(net_drop.parameters(), lr=LR)
loss_func = nn.MSELoss()

plt.ion()       # 打开交互模式，在交互模式下，plt.plot()或plt.imshow()是直接出图像，不需要plt.show()

for t in range(EPOCH):
    pred_normal = net_normal(x)
    pred_drop = net_drop(x)
    loss_normal = loss_func(pred_normal, y)
    loss_drop = loss_func(pred_drop, y)

    optimizer_normal.zero_grad()
    loss_normal.backward()
    optimizer_normal.step()

    optimizer_drop.zero_grad()
    loss_drop.backward()
    optimizer_drop.step()

    if t % 10 == 0:
        net_drop.eval()                             # 把包含drop的神经网络转为预测模式，因为在预测模式下，它会注释掉Dropout层，展示整个神经网络的效果

        plt.cla()                                   # 清除图像中的活动轴，其他轴不受影响，用于刷新绘图
        test_pred_normal = net_normal(test_x)
        test_pred_drop = net_drop(test_x)
        plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5, label='train')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label='test')
        plt.plot(test_x.data.numpy(), test_pred_normal.data.numpy(), c='r', lw=3, label='normal')
        plt.plot(test_x.data.numpy(), test_pred_drop.data.numpy(), c='b', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'normal_loss=%.4f' % loss_func(test_pred_normal, test_y).data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.text(0,-1.5, 'dropout_loss=%.4f' % loss_func(test_pred_drop, test_y).data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.legend(loc='best')
        plt.ylim((-2.5, 2.5))
        plt.pause(0.1)

        net_drop.train()                            # 把包含drop的神经网络转为训练模式，让之后的训练不受影响

plt.ioff()
plt.show()
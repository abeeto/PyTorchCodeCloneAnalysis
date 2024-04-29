import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
from visdom import Visdom

viz = Visdom()

# 데이터 수를 늘리니까 viz가 죽음..
num_data = 5000  # 데이터 수를 늘리면 더 잘되지 않을까?
num_epoch = 20000  # 러닝 횟수를 더 높이면 러닝이 더 잘되지 않을까? 안될거 같긴하다...
MARKER_SIZE = 1

noise = init.normal(torch.FloatTensor(num_data, 1), std=1)

x = init.uniform(torch.Tensor(num_data, 1), -15, 15)
y = (x**2) + 3

y_noise = y + noise
# 일단, noise가 없으면, 러닝이 더 잘되지 않을까? 라는 게 가설.
# 그런데 이러니까 성능평가가 안됨... 그래서 다시 노이즈를 쓰는 식으로 테스트.

# visualize data
input_data = torch.cat([x, y_noise], 1)

win = viz.scatter(
    X=input_data,
    opts=dict(
        xtickmin=-15,
        xtickmax=15,
        xtickstep=1,
        ytickmin=0,
        ytickmax=500,
        ytickstep=1,
        markersymbol='dot',
        markercolor=np.random.randint(0, 255, num_data),
        markersize=MARKER_SIZE, ), )

viz.updateTrace(
    X=x,
    Y=y,
    win=win, )

# fully connected model with 4 hidden layer

model = nn.Sequential(
    nn.Linear(1, 6),
    nn.ReLU(),
    nn.Linear(6, 10),
    nn.ReLU(),
    nn.Linear(10, 6),
    nn.ReLU(),
    nn.Linear(6, 1), ).cuda()

loss_func = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.0005)

loss_arr = []
label = Variable(y_noise.cuda())

for i in range(num_epoch):
    optimizer.zero_grad()
    output = model(Variable(x.cuda()))

    loss = loss_func(output, label)
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(loss)

    loss_arr.append(loss.cpu().data.numpy()[0])

param_list = list(model.parameters())
print(param_list)

win2 = viz.scatter(
    X=input_data,
    opts=dict(
        xtickmin=-15,
        xtickmax=10,
        xtickstep=1,
        ytickmin=0,
        ytickmax=500,
        ytickstep=1,
        markersymbol='dot',
        markercolor=np.random.randint(0, 255, num_data),
        markersize=MARKER_SIZE, ), )

viz.updateTrace(
    X=x,
    Y=output.cpu().data,
    win=win2, )

x = np.reshape([i for i in range(num_epoch)], newshape=[num_epoch, 1])
loss_data = np.reshape(loss_arr, newshape=[num_epoch, 1])

win3 = viz.line(
    X=x,
    Y=loss_data, )

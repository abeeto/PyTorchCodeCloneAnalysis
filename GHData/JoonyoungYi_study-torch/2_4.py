import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable

# visdom is a visualization tool from facebook
from visdom import Visdom
viz = Visdom()

num_data = 1000
num_epoch = 1000

noise = init.normal(torch.FloatTensor(num_data, 1), std=1)
x = init.uniform(torch.Tensor(num_data, 1), -15, 10)
y = -x**3 - 8 * (x**2) + 3

y_noise = y + noise

# visualize data with visdom

input_data = torch.cat([x, y_noise], 1)

win = viz.scatter(
    X=input_data,
    opts=dict(
        xtickmin=-15,
        xtickmax=10,
        xtickstep=1,
        ytickmin=-300,
        ytickmax=200,
        ytickstep=1,
        markersymbol='dot',
        markercolor=np.random.randint(0, 255, num_data),
        markersize=5, ), )

viz.updateTrace(
    X=x,
    Y=y,
    win=win, )

model = nn.Linear(1, 1)
output = model(Variable(x))

loss_func = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# train
loss_arr = []
label = Variable(y_noise)
for i in range(num_epoch):
    output = model(Variable(x))
    optimizer.zero_grad()

    loss = loss_func(output, label)
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(loss)
    loss_arr.append(loss.data.numpy()[0])

param_list = list(model.parameters())
print(param_list[0].data, param_list[1].data)

win_2 = viz.scatter(
    X=input_data,
    opts=dict(
        xtickmin=-15,
        xtickmax=10,
        xtickstep=1,
        ytickmin=-300,
        ytickmax=200,
        ytickstep=1,
        markersymbol='dot',
        markercolor=np.random.randint(0, 255, num_data),
        markersize=5, ), )

viz.updateTrace(
    X=x,
    Y=output.data,
    win=win_2,
    opts=dict(
        xtickmin=-15,
        xtickmax=10,
        xtickstep=1,
        ytickmin=-300,
        ytickmax=200,
        ytickstep=1,
        markersymbol='dot', ), )

x = np.reshape([i for i in range(num_epoch)], newshape=[num_epoch, 1])
loss_data = np.reshape(loss_arr, newshape=[num_epoch, 1])

win2 = viz.line(
    X=x,
    Y=loss_data, )

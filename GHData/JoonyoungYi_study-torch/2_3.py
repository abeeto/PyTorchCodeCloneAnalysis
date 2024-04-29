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
x = init.uniform(torch.Tensor(num_data, 1), -10, 10)

y = 2 * x + 3
y_noise = 2 * x + 3 + noise

# visualize data with visdom
input_data = torch.cat([x, y_noise], 1)
# win = viz.scatter(
#     X=input_data,
#     opts=dict(
#         xtickmin=-10,
#         xtickmax=10,
#         xtickstep=1,
#         ytickmin=-20,
#         ytickmax=20,
#         ytickstep=1,
#         markersymbol='dot',
#         markersize=5,
#         markercolor=np.random.randint(0, 255, num_data), ), )
#
# viz.updateTrace(
#     X=x,
#     Y=y,
#     win=win, )

model = nn.Linear(1, 1)
output = model(Variable(x))
loss_func = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# train
loss_arr = []
label = Variable(y_noise)
# print(label)
for i in range(num_epoch):
    optimizer.zero_grad()
    output = model(Variable(x))

    loss = loss_func(output, label)
    loss.backward()
    optimizer.step()

    if loss.data[0] < 1:
        break

    if i % 10 == 0:
        param_list = list(model.parameters())
        print(">>", loss.data[0], param_list[0].data[0][0],
              param_list[1].data[0])

    loss_arr.append(loss.data.numpy()[0])
    # print(label)

param_list = list(model.parameters())
print(param_list[0].data, param_list[1].data)

win_2 = viz.scatter(
    X=input_data,
    opts=dict(
        xtickmin=-10,
        xtickmax=10,
        xtickstep=1,
        ytickmin=-20,
        ytickmax=20,
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

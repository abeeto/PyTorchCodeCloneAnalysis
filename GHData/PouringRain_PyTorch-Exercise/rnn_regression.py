# encoding=utf8
# input_x: sinx  predict_y: cosx

import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)

TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.01

#show pic
# step = np.linspace(0, 2*np.pi, 100, dtype=np.float32)
# # print(step)
# x = np.sin(step)
# y = np.cos(step)
# plt.plot(step, x, "r-", label='input_x')
# plt.plot(step, y, 'b-', label='target_y')
# plt.legend(loc='best')
# plt.show()

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            hidden_size=32,
            input_size=1,
            batch_first=True,
            num_layers=1
        )

        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        outs = [] # store outputs

        for step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, step, :]))

        return torch.stack(outs, dim=1), h_state

rnn = RNN()
print(rnn)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

h_state = None
plt.figure(1, figsize=(12, 5))
plt.ion()

for step in range(60):

    start, end = step*np.pi, (step+1)*np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)


    x = Variable(torch.from_numpy(x_np[np.newaxis, : ,np.newaxis]))
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))


    predict_y, h_state = rnn(x, h_state)
    h_state = Variable(h_state.data)

    loss = loss_func(predict_y, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, predict_y.data.numpy().flatten(), 'b-')
    plt.draw(); plt.pause(0.05)

plt.ioff()
plt.show()
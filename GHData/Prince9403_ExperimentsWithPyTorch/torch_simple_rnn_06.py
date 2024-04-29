import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

"""
thor_review = "the action scenes were top notch in this movie. \
            Thor has never been this epic in the MCU. He does some pretty epic shot \
            in this movie and he is definitely not uner-powered anymore. \
            Thor is unleashed in this, I love that."
"""

x = np.array([2, 3 ,5])
x = np.reshape(x, (3, 1, 1))

thor_review = torch.from_numpy(x).float()

input_size = 14
hidden_size = 8
output_size = 15

rnn = RNN(input_size, hidden_size, output_size)
hidden = rnn.initHidden()
for i in range(len(thor_review)):
    output, hidden = rnn(thor_review[i], hidden)

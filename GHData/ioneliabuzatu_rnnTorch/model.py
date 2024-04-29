import torch.nn as nn
import torch


class RNN(nn.Module):
    """
    :net 2 linear layers which operate on an input and hidden state, with a LogSoftmax layer after the output.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(input_size+ hidden_size, hidden_size)
        self.linear2 = nn.Linear(input_size + hidden_size, output_size) # not sure the architecture of this
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.linear1(combined)
        output = self.linear2(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


if __name__ == '__main__':
    rnn = RNN()

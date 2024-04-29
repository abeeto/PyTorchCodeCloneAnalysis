import torch
import torch.nn as nn
from RNNClassification import unicodetoascii, n_letters, nametotensor


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):

        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(1)

        hidden = self.init_hidden(batch_size)
        cell = self.cell_state(batch_size)

        output, hidden = self.lstm(x, (hidden, cell))

        last_output = output[-1]  # batch_size * hidden_size

        fc_out = self.fc(last_output)  # 1 * hidden_size

        return fc_out

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

    def cell_state(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


def main():
    n_hidden = 128
    n_categories = 2
    lstm = LSTM(n_letters, n_hidden, n_categories, num_layers=2)
    print(lstm)

    name = nametotensor(unicodetoascii('Albert'))
    print(name)
    print(name.size())

    fcout = lstm.forward(name)
    print(fcout)
    print(fcout.size())


if __name__ == '__main__':
    main()


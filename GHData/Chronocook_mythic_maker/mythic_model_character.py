"""
This class defines the model setup for a character level
RNN in the torch framework
Credit goes to: https://github.com/spro/char-rnn.pytorch
that HIGHLY influenced this code, thanks!

@author: Brad Beechler (brad.e.beechler@gmail.com)
# Last Modification: 09/20/2017 (Brad Beechler)
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model="gru", cuda=None,
                 n_layers=1, dropout=0.2):
        super(CharRNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.cuda = cuda

        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.model == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.rnn.dropout = dropout
        if cuda is not None:
            self.encoder.cuda()
            self.rnn.cuda()
            self.decoder.cuda()

    def forward(self, input_pattern, hidden):
        batch_size = input_pattern.size(0)
        encoded = self.encoder(input_pattern)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def forward2(self, input_pattern, hidden):
        encoded = self.encoder(input_pattern.view(1, -1))
        output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self, batch_size, cuda=None):
        if cuda is not None:
            if self.model == "lstm":
                return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda(device=cuda)),
                        Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda(device=cuda)))
            return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda(device=cuda))
        else:
            if self.model == "lstm":
                return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                        Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
            return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

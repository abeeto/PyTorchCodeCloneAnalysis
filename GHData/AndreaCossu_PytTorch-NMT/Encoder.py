import torch
import torch.nn as nn
from torch.autograd import Variable


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, pars, n_layers=1):
        super(EncoderRNN, self).__init__()

        # Encoder Parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.pars = pars

        # Defining Encoder layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, word_inputs, hidden):
        # Applying Encoder's model
        # Note: we run this all at once (over the whole input sequence)
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        # Encoder's hidden layer initialization
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if self.pars.USE_CUDA:
            hidden = hidden.cuda()

        return hidden

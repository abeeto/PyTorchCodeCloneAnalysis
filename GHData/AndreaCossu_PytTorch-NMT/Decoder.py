import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Attn(nn.Module):
    def __init__(self, hidden_size, pars):
        super(Attn, self).__init__()

        self.hidden_size = hidden_size
        self.pars = pars

        self.attn = nn.Linear(self.hidden_size, hidden_size)

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        # Creating variable to store attention energies
        if self.pars.USE_CUDA:
            attn_energies = Variable(torch.zeros(seq_len)).to(torch.device("cuda"))  # B x 1 x S
        else:
            attn_energies = Variable(torch.zeros(seq_len))

        # Calculating energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalizing energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)

    def score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        energy = hidden.squeeze(0).dot(energy.squeeze(0)).unsqueeze(0)
        return energy


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, pars, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        # Decoder parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Defining Decoder layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

        # Choosing attention model
        self.attn = Attn(hidden_size, pars)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # Applying Decoder's model
        # Note: we run this one step at a time

        # Getting the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1)  # S=1 x B x N

        # Combining embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculating attention from current RNN state and all encoder outputs; applying to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)        # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))

        # Returning final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights

import torch
import torch.nn as nn
import numpy as np


# two interesting autoencoders may come handy:
# Sequence-to-sequence autoencoder(i.g. uses LSTM) and Variational autoencoder (VAE)
class autoencoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(autoencoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstf = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # weights
        nn.init.xavier_uniform(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.lstm.weight_ih_l0, gain=np.sqrt(2))

    def encoder(self, input):
        tt = torch.cuda if self.isCuda else torch
        h0 = torch.Variable(tt.FloatTensor(self.num_layers, input.size(0), self.hidden_size))
        c0 = torch.Variable(tt.FloatTensor(self.num_layers, input.size(0), self.hidden_size))
        encoded_input, hidden = self.lstm(input, (h0, c0))
        encoded_input = self.relu(encoded_input)
        return encoded_input


    def decoder(self, encoded_input):
        tt = torch.cuda if self.isCuda else torch
        h0 = torch.Variable(tt.FloatTensor(self.num_layers, encoded_input.size(0), self.output_size))
        c0 = torch.Variable(tt.FloatTensor(self.num_layers, encoded_input.size(0), self.output_size))
        decoded_output, hidden = self.lstm(encoded_input, (h0, c0))
        decoded_output = self.sigmoid(decoded_output)
        return decoded_output

    def lstm(self):
        self.encoder = self.encoder()
        pass
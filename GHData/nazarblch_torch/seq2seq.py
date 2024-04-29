# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.gru = nn.LSTM(input_size, hidden_size)

    def forward(self, input, hidden):
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        return output[0, 0], hidden

    def encode(self, input_seq):
        hidden = self.initHidden()
        for i in range(len(input_seq)):
            input = input_seq[i]
            output, hidden = self.forward(input, hidden)
        return hidden

    def initHidden(self):
        return  (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.LSTM(output_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)


    def forward(self, input, hidden):
        output = input.view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0, 0])
        return output, hidden

    def decode(self, input, target_length):
        hidden = input
        output = torch.zeros(1, device=device)
        res = []
        for di in range(target_length):
            t = torch.tensor([float(di)], device=device).reshape(1)
            output, hidden = self.forward(t, hidden)
            res.append(output)

        return torch.cat(res)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



class Seq2SeqModel(nn.Module):

    def __init__(self, size):
        super(Seq2SeqModel, self).__init__()
        self.size = size

        self.encoder = EncoderRNN(1, size)
        self.decoder = DecoderRNN(size, 1)

        self.encoder_optimizer = optim.Adagrad(self.encoder.parameters(), lr=0.3)
        self.decoder_optimizer = optim.Adagrad(self.decoder.parameters(), lr=0.3)

        self.loss_fn = torch.nn.MSELoss(reduction="sum")

    def weighted_mse(self, y1, y2, weights):
        lseq = (y1 - y2) ** 2
        return lseq.view(len(y1)).dot(weights)


    def train(self, data, steps, weights=None):

        loss = 0
        y_pred = None

        n = len(data)

        if weights is None:
            weights = torch.ones(n, dtype=torch.float)
        else:
            weights = torch.tensor(weights, dtype=torch.float)

        y = torch.tensor(data, dtype=torch.float).reshape(n)

        for t in range(steps):

            info = self.encoder.encode(y)
            y_pred = self.decoder.decode(info, n).reshape(n)
            # l1_regularization = 0.01 * torch.norm(info, 1)
            # loss = self.loss_fn(y_pred, y)
            loss = self.weighted_mse(y_pred, y, weights)
            l1loss = loss  # + l1_regularization

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            l1loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            print(loss)

        return loss, y_pred
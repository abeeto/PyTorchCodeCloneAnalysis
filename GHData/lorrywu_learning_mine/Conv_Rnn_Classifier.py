from __future__ import print_function
import numpy as np
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, hidden_size, batch_size, use_cuda, vocabulary_size, embedding_size, output_size, max_sentence_len):
        super(Net, self).__init__()

        self.use_cuda = use_cuda
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.max_sentence_len = max_sentence_len + 10

        self.embed = nn.Embedding(self.vocabulary_size, self.embedding_size)
        self.LSTM_cell_enc_f = nn.LSTMCell(input_size=self.embedding_size, hidden_size=self.hidden_size)
        self.LSTM_cell_enc_b = nn.LSTMCell(input_size=self.embedding_size, hidden_size=self.hidden_size)

        self.cnn = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2)


        self.fc_0 = nn.Linear(in_features=self.hidden_size+self.hidden_size+(self.hidden_size/2), out_features=self.hidden_size)
        self.fc_1 = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

    def forward(self, x):

        input_enc_h0 = Variable(torch.zeros((self.batch_size, self.hidden_size))).float()
        input_enc_h0 = input_enc_h0.cuda() if self.use_cuda else input_enc_h0

        input_enc_c0 = Variable(torch.zeros((self.batch_size, self.hidden_size))).float()
        input_enc_c0 = input_enc_c0.cuda() if self.use_cuda else input_enc_c0

        input_enc_h1 = Variable(torch.zeros((self.batch_size, self.hidden_size))).float()
        input_enc_h1 = input_enc_h1.cuda() if self.use_cuda else input_enc_h1

        input_enc_c1 = Variable(torch.zeros((self.batch_size, self.hidden_size))).float()
        input_enc_c1 = input_enc_c1.cuda() if self.use_cuda else input_enc_c1

        #before x is batch * seqlen * features
        x = torch.transpose(x, 0, 1)
        #now x is seqlen * batch * features


        forward_rnn_code = []

        # forward lstm
        for i in range(0, len(x)):
            i = self.embed(x[i])
            i = torch.squeeze(i, dim=0)
            input_enc_h0, input_enc_c0 = self.LSTM_cell_enc_f(i, (input_enc_h0, input_enc_c0))
            forward_rnn_code.append(input_enc_h0)

        forward_last_state = torch.squeeze(input_enc_h0)

        backward_rnn_code = []

        # backward lstm
        for i in range(len(x)-1, -1, -1):
            i = self.embed(x[i])
            i = torch.squeeze(i, dim=0)
            input_enc_h1, input_enc_c1 = self.LSTM_cell_enc_b(i, (input_enc_h1, input_enc_c1))
            backward_rnn_code.append(input_enc_h1)

        backward_last_state = torch.squeeze(input_enc_h1)

        concate_rnn_code = []
        # concatenate forward and backward rnn codes
        for i in range(len(x)):
            concate = torch.cat((forward_rnn_code[i], backward_rnn_code[i]), dim=1)
            concate = torch.squeeze(concate, dim=0)
            concate_rnn_code.append(concate)
        # make concate_rnn_code to tensor
        concate_rnn_code = torch.stack(concate_rnn_code)
        concate_rnn_code = torch.unsqueeze(concate_rnn_code, dim=0)
        concate_rnn_code = torch.unsqueeze(concate_rnn_code, dim=0)

        # padding cnn input
        pad_dim = self.max_sentence_len - concate_rnn_code.size()[2]
        pad = Variable(torch.zeros((1, 1, pad_dim, concate_rnn_code.size()[3])).float())
        pad = pad.cuda() if self.use_cuda else pad
        concate_rnn_code = torch.cat((concate_rnn_code, pad), dim=2)


        conv2d_rnn_code = self.cnn(concate_rnn_code)

        max_pooled = F.max_pool2d(input=conv2d_rnn_code, kernel_size=(conv2d_rnn_code.size()[2], 2), stride=2)

        max_pooled = torch.squeeze(max_pooled)

        combined = torch.cat((forward_last_state, max_pooled, backward_last_state), dim=0)
        combined = torch.unsqueeze(combined, dim=0)
        output = self.fc_0(combined)
        output = self.fc_1(output)

        return output
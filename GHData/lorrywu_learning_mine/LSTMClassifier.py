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

        self.embed = nn.Embedding(self.vocabulary_size, self.embedding_size)
        self.LSTM_cell_enc_0 = nn.LSTMCell(input_size=self.embedding_size, hidden_size=self.hidden_size)

        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)


    def forward(self, x):

        input_enc_h0 = Variable(torch.zeros((self.batch_size, self.hidden_size))).float()
        input_enc_h0 = input_enc_h0.cuda() if self.use_cuda else input_enc_h0

        input_enc_c0 = Variable(torch.zeros((self.batch_size, self.hidden_size))).float()
        input_enc_c0 = input_enc_c0.cuda() if self.use_cuda else input_enc_c0

        #before x is batch * seqlen * features
        x = torch.transpose(x, 0, 1)
        #now x is seqlen * batch * features

        # encoder start
        for i in x:
            i = self.embed(i)
            i = torch.squeeze(i, dim=0)
            input_enc_h0, input_enc_c0 = self.LSTM_cell_enc_0(i, (input_enc_h0, input_enc_c0))

        output = self.fc(input_enc_h0)

        return output

# concat CNN and LSTM output

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random

torch.cat(torch.tensor[[1,1,1],[1,1,1],[1,1,1]],1)


class CNN_LSTM(nn.Module):
    
    def __init__(self, opts):
        super(CNN_LSTM, self).__init__()
        
        random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)
        
        self.opts = opts
        # emmbeding parameter
        V = opts.embed_num # vocab size
        D = opts.embed_dim # embeding_size
        
        # CNN parameter
        C = opts.class_num
        Ci = 1
        Co = opts.kernel_num
        Ks = opts.kernel_sizes    
        # LSTM parameter
        self.hidden_dim = opts.lstm_hidden_dim
        self.num_layers = opts.lstm_num_layers
        
        self.C = C 
        self.embed = nn.Embedding(V, D, padding_idx=opts.paddingId)
        
        # pretrained  embedding
        if opts.word_Embedding:
            self.embed.weight.data.copy_(opts.pretrained_weight)

        # CNN
        self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.dropout = nn.Dropout(opts.dropout)
        
        # for cnn cuda
        if self.opts.cuda is True:
            for conv in self.convs1:
                conv = conv.cuda()

        # LSTM
        self.lstm = nn.LSTM(D, self.hidden_dim, dropout=opts.dropout, num_layers=self.num_layers)

        # linear
        L = len(Ks) * Co + self.hidden_dim
        self.fc1 = nn.Linear(L, L // 2)
        self.fc2 = nn.Linear(L // 2, C)

    def forward(self, x):
        embed = self.embed(x)

        # CNN
        cnn_x = embed
        cnn_x = torch.transpose(cnn_x, 0, 1)
        cnn_x = cnn_x.unsqueeze(1)
        cnn_x = [F.relu(conv(cnn_x)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        cnn_x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn_x]  # [(N,Co), ...]*len(Ks)
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = self.dropout(cnn_x)

        # LSTM
        lstm_x = embed.view(len(x), embed.size(1), -1)
        lstm_out, _ = self.lstm(lstm_x)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)

        # CNN and LSTM cat
        cnn_x = torch.transpose(cnn_x, 0, 1)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        cnn_lstm_out = torch.cat((cnn_x, lstm_out), 0)
        cnn_lstm_out = torch.transpose(cnn_lstm_out, 0, 1)

        # linear
        cnn_lstm_out = self.fc1(F.tanh(cnn_lstm_out))
        cnn_lstm_out = self.fc2(F.tanh(cnn_lstm_out))

        # output
        logit = cnn_lstm_out
        return logit
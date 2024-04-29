# Highway LSTM (HLSTM)

# https://arxiv.org/abs/1510.08983


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
import random



class HighWay_BiLSTM(nn.Module):
    def __init__(self, opts):
        super(HighWay_BiLSTM_1, self).__init__()
        
        random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)
     
        self.opts = opts
        
        # network parameter
        self.hidden_dim = opts.lstm_hidden_dim
        self.num_layers = opts.lstm_num_layers
        
        # embedding parameter
        V = opts.embed_num
        D = opts.embed_dim
        
        C = opts.class_num
        
        self.dropout = nn.Dropout(opts.dropout)
            
        self.embed = nn.Embedding(V, D, padding_idx=opts.paddingId)
        # pretrained  embedding
        if opts.word_Embedding:
            self.embed.weight.data.copy_(opts.pretrained_weight)

        self.bilstm = nn.LSTM(D, self.hidden_dim, 
                              num_layers=self.num_layers, 
                              bias=True, 
                              bidirectional=True,
                              dropout=self.opts.dropout)

        if opts.init_weight:
            
            init.xavier_normal(self.bilstm.all_weights[0][0], gain=np.sqrt(opts.init_weight_value))
            init.xavier_normal(self.bilstm.all_weights[0][1], gain=np.sqrt(opts.init_weight_value))
            init.xavier_normal(self.bilstm.all_weights[1][0], gain=np.sqrt(opts.init_weight_value))
            init.xavier_normal(self.bilstm.all_weights[1][1], gain=np.sqrt(opts.init_weight_value))

            # init weight of lstm gate
            self.bilstm.all_weights[0][3].data[20:40].fill_(1)
            self.bilstm.all_weights[0][3].data[0:20].fill_(0)
            self.bilstm.all_weights[0][3].data[40:80].fill_(0)
            # self.bilstm.all_weights[0][3].data[40:].fill_(0)
            self.bilstm.all_weights[0][2].data[20:40].fill_(1)
            self.bilstm.all_weights[0][2].data[0:20].fill_(0)
            self.bilstm.all_weights[0][2].data[40:80].fill_(0)
            # self.bilstm.all_weights[0][2].data[40:].fill_(0)
            self.bilstm.all_weights[1][3].data[20:40].fill_(1)
            self.bilstm.all_weights[1][3].data[0:20].fill_(0)
            self.bilstm.all_weights[1][3].data[40:80].fill_(0)
            # self.bilstm.all_weights[1][3].data[40:].fill_(0)
            self.bilstm.all_weights[1][2].data[20:40].fill_(1)
            self.bilstm.all_weights[1][2].data[0:20].fill_(0)
            self.bilstm.all_weights[1][2].data[40:80].fill_(0)
            # self.bilstm.all_weights[1][2].data[40:].fill_(0)

        self.fc1 = nn.Linear(in_features=self.hidden_dim * 2, out_features=self.hidden_dim * 2, bias=True)

        # highway gate layer
        self.gate_layer = nn.Linear(in_features=self.hidden_dim * 2, out_features=self.hidden_dim * 2, bias=True)

        # last liner
        self.logit_layer = nn.Linear(in_features=self.hidden_dim * 2, out_features=C, bias=True)

    def forward(self, x):
        x = self.embed(x)
        x = self.dropout(x)
        bilstm_out, _ = self.bilstm(x)

        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2))
        bilstm_out = bilstm_out.squeeze(2)

        hidden2lable = self.fc1(F.tanh(bilstm_out))

        gate_layer = F.sigmoid(self.gate_layer(bilstm_out))
        
        # calculate highway layer values
        gate_hidden_layer = torch.mul(hidden2lable, gate_layer)
        gate_input = torch.mul((1 - gate_layer), bilstm_out)
        highway_output = torch.add(gate_hidden_layer, gate_input)

        logit = self.logit_layer(highway_output)

        return logit
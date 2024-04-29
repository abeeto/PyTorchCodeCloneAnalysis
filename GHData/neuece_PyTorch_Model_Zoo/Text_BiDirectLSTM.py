# Bidirectional LSTM


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.init as init
from DataUtils.Common import seed_num





class BiLSTM_1(nn.Module):
    def __init__(self, opts):
        super(BiLSTM_1, self).__init__()
        
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
        # dropout
        self.dropout = nn.Dropout(opts.dropout)
        self.dropout_embed = nn.Dropout(opts.dropout_embed)
        
        
        if opts.max_norm is not None:
            self.embed = nn.Embedding(V, D, max_norm=opts.max_norm, scale_grad_by_freq=True, padding_idx=opts.paddingId)
            # pretrained  embedding
            if opts.word_Embedding:
                self.embed.weight.data.copy_(opts.pretrained_weight)
        else:
            self.embed = nn.Embedding(V, D, scale_grad_by_freq=True, padding_idx=opts.paddingId)
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

        self.fc = nn.Linear(self.hidden_dim * 2, C)

    def forward(self, x):
        x = self.embed(x)
        x = self.dropout_embed(x)
        bilstm_out, _ = self.bilstm(x)

        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        bilstm_out = F.tanh(bilstm_out)

        logit = self.fc(bilstm_out)

        return logit

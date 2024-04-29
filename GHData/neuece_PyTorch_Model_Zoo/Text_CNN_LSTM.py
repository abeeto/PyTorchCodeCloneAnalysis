# CNN then LSTM

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import random




class CLSTM(nn.Module):
    
    def __init__(self, opts):
        super(CLSTM, self).__init__()
        
        random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)
        
        self.opts = opts
        
        # LSTM network parameter
        self.hidden_dim = opts.lstm_hidden_dim
        self.num_layers = opts.lstm_num_layers
        
        # embedding parameter
        V = opts.embed_num
        D = opts.embed_dim
        C = opts.class_num
        
        # CNN network parameter
        Ci = 1
        Co = opts.kernel_num
        Ks = opts.kernel_sizes
        
        # embedding
        self.embed = nn.Embedding(V, D, padding_idx=opts.paddingId)
        # pretrained  embedding
        if opts.word_Embedding:
            self.embed.weight.data.copy_(opts.pretrained_weight)

        # CNN
        LK = []
        for K in Ks:
            LK.append( K + 1 if K % 2 == 0 else K)
#        self.convs1 = [nn.Conv2d(Ci, Co, (K, D), stride=1, padding=(K//2, 0)) for K in KK]
        self.convs1 = [nn.Conv2d(Ci, D, (K, D), stride=1, padding=(K//2, 0)) for K in LK]
            
        # LSTM
        self.lstm = nn.LSTM(D, self.hidden_dim, num_layers= self.num_layers, dropout=opts.dropout)

        # linear
        self.hidden2label1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.hidden2label2 = nn.Linear(self.hidden_dim // 2, C)
        
        # dropout
        self.dropout = nn.Dropout(opts.dropout)
        
        # send cnn to cuda
        if self.opts.cuda is True:
            for conv in self.convs1:
                conv = conv.cuda()

    def forward(self, x):
        embed = self.embed(x)
        
        # CNN
        cnn_x = embed
        cnn_x = self.dropout(cnn_x)
        cnn_x = cnn_x.unsqueeze(1)
        cnn_x = [F.relu(conv(cnn_x)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        cnn_x = torch.cat(cnn_x, 0)
        cnn_x = torch.transpose(cnn_x, 1, 2)
        
        # LSTM
        # Using CNN output as input of LSTM
        lstm_out, _ = self.lstm(cnn_x)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        
        # linear
        cnn_lstm_out = self.hidden2label1(F.tanh(lstm_out))
        cnn_lstm_out = self.hidden2label2(F.tanh(cnn_lstm_out))
        
        # output
        logit = cnn_lstm_out

        return logit
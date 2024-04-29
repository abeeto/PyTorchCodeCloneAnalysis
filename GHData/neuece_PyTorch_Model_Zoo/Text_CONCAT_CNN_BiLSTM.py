# Bidirectional LSTM + CNN


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random


"""
    Neural Network: CNN_BiLSTM
    Detail: the input crosss cnn model and LSTM model independly, then the result of both concat
"""


class CNN_BiLSTM(nn.Module):

    def __init__(self, opts):
        super(CNN_BiLSTM, self).__init__()
        
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
        
        # Embeddings
        self.embed = nn.Embedding(V, D, padding_idx=opts.paddingId)
        # pretrained  embedding
        if opts.word_Embedding:
            self.embed.weight.data.copy_(opts.pretrained_weight)
        
        
        # CNN network parameter
        Ci = 1
        Co = opts.kernel_num
        Ks = opts.kernel_sizes
        
        C = opts.class_num
        self.C = C

        # CNN
        self.convs1 = [nn.Conv2d(Ci, Co, (K, D), padding=(K//2, 0), stride=1) for K in Ks]
 
        # for cnn cuda
        if self.opts.cuda is True:
            for conv in self.convs1:
                conv = conv.cuda()

        # BiLSTM
        self.bilstm = nn.LSTM(D, self.hidden_dim, num_layers=self.num_layers, dropout=opts.dropout, bidirectional=True, bias=True)

        # linear
        L = len(Ks) * Co + self.hidden_dim * 2
        self.fc1 = nn.Linear(L, L // 2)
        self.fc2 = nn.Linear(L // 2, C)

        # dropout
        self.dropout = nn.Dropout(opts.dropout)

    def forward(self, x):
        embed = self.embed(x)

        # CNN
        cnn_x = embed
        cnn_x = torch.transpose(cnn_x, 0, 1)
        cnn_x = cnn_x.unsqueeze(1)
        cnn_x = [conv(cnn_x).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        cnn_x = [F.tanh(F.max_pool1d(i, i.size(2)).squeeze(2)) for i in cnn_x]  # [(N,Co), ...]*len(Ks)
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = self.dropout(cnn_x)

        # BiLSTM
        bilstm_x = embed.view(len(x), embed.size(1), -1)
        bilstm_out, _ = self.bilstm(bilstm_x)
        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        bilstm_out = F.tanh(bilstm_out)

        # Concat CNN and BiLSTM 
        cnn_x = torch.transpose(cnn_x, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        cnn_bilstm_out = torch.cat((cnn_x, bilstm_out), 0)
        cnn_bilstm_out = torch.transpose(cnn_bilstm_out, 0, 1)

        # fc layers
        cnn_bilstm_out = self.fc1(F.tanh(cnn_bilstm_out))
        cnn_bilstm_out = self.fc2(F.tanh(cnn_bilstm_out))

        # output
        logit = cnn_bilstm_out
        return logit
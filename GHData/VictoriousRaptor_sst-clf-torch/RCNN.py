# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class RCNN(nn.Module):
    def __init__(self, config):
        super(RCNN, self).__init__()

        self.label_num = config.label_num
        self.hidden_size = 100  
        self.hidden_size_linear = 64  # dim after pooling
        self.embedding_length = config.wordvec_dim
        self.device = config.device
        self.embeddings = nn.Embedding(config.weight.shape[0], self.embedding_length).to(config.device)
        self.embeddings = self.embeddings.from_pretrained(config.weight, freeze=False)
        self.lstm = nn.LSTM(input_size = self.embedding_length,
                            hidden_size = self.hidden_size,
                            num_layers = 1,
                            bidirectional = True, batch_first=True)
        
        self.dropout = nn.Dropout(0.2)  # 20% to be zeroed
        
        self.W = nn.Linear(2*self.hidden_size+self.embedding_length, self.hidden_size_linear)
        # self.W = nn.Linear(self.hidden_size+self.embedding_length, self.hidden_size_linear)
        
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(self.hidden_size_linear, config.label_num)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        embedded_sent = self.embeddings(x)  # (batch_size, seq_len, embed_size)

        lstm_out, (h_n, c_n) = self.lstm(embedded_sent)  # (batch_size, seq_len, 2 * hidden_size)

        input_features = torch.cat([lstm_out, embedded_sent], 2)  # (batch_size, seq_len, embed_size + 2*hidden_size)

        # The method described in the original paper, very slow
        # maybe torch.split?
        # input_features = torch.zeros((x.size()[0], x.size()[1], self.hidden_size+self.embedding_length), device=self.device)
        # for j in range(x.size()[1]):
        #     for h in range(self.hidden_size):
        #         input_features[:, j, :] = torch.cat([lstm_out[:, j, :h], embedded_sent[:, j, :], lstm_out[:, j, h-self.hidden_size:]], dim=1)
        # input_features = torch.zeros((x.size()[0], x.size()[1], self.hidden_size+self.embedding_length), device=self.device)
        # for h in range(self.hidden_size):
        #     input_features[:, :, :] = torch.cat([lstm_out[:, :, :h], embedded_sent[:, :, :], lstm_out[:, :, h-self.hidden_size:]], dim=2)

        
        linear_output = self.tanh(self.W(input_features))  # (batch_size, seq_len, hidden_size_linear)
        
        linear_output = linear_output.permute(0,2,1)
        
        max_out_features = F.max_pool1d(linear_output, linear_output.shape[2]).squeeze(2)  # (batch_size, hidden_size_linear)
        
        # max_out_features = self.dropout(max_out_features)
        final_out = self.fc(max_out_features)
        # return F.softmax(final_out, dim=1)
        return final_out



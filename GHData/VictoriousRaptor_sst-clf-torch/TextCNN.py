# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TextCNN(nn.Module):
    
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.kernel_num = config.kernel_num  # Number of conv kernels
        self.embedding_length = config.wordvec_dim  
        self.Ks = config.kernel_sizes
        self.word_embeddings = nn.Embedding(config.weight.shape[0], self.embedding_length)  # Embedding layer
        self.word_embeddings = self.word_embeddings.from_pretrained(config.weight, freeze=False)  # Load pretrianed word embedding, and fine-tuing
        
        self.convs = nn.ModuleList([nn.Conv2d(1, config.kernel_num, (K, config.wordvec_dim), bias=True) for K in self.Ks])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(self.Ks)*self.kernel_num, config.label_num)
        self.bn1 = nn.BatchNorm1d(len(self.Ks)*self.kernel_num)
        # self.fc1 = nn.Linear(len(self.Ks)*self.kernel_num, fc1out)
        # self.bn2 = nn.BatchNorm1d(fc1out)
        # self.fc2 = nn.Linear(fc1out, config.label_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # (batch_size, batch_dim)
        # Embedding
        x = self.word_embeddings(x)  # (batch_size, batch_dim, embedding_len)

        x = x.unsqueeze(1)  # (batch_size, 1, batch_dim, embedding_len)
        # print(x.size())

        # Conv and relu
        x = [self.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(batch_size, kernel_num, batch_dim-K+1), ...]*len(Ks)
        # print([i.shape for i in x])
        
        # max-over-time pooling (actually merely max)
        x = [F.max_pool1d(xi, xi.size()[2]).squeeze(2) for xi in x] # [(batch_size, kernel_num), ...]*len(Ks)
        # print([i.shape for i in x])
        x = torch.cat(x, dim=1)  # (batch_size, kernel_num*len(Ks))
        

        x = self.bn1(x)  # Batch Normaliztion
        x = self.dropout(x)  # (batch_size, len(Ks)*kernel_num), dropout
        # print(x.size())

        x = self.fc(x)  # (batch_size, label_num)
        # print(x.size())

        # x = self.fc1(x)
        # x = self.bn2(x)
        # x = self.fc2(x)
        logit = self.softmax(x)
        return logit

if __name__ == '__main__':
    pass

# CNN with PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# self defined utils
import utils.Embedding as Embedding


class CNN(nn.Module):

    def __init__(self, opts, vocab, label_vocab):
        super(CNN, self).__init__()

        random.seed(opts.seed)
        torch.cuda.manual_seed(opts.gpu_seed)
        
        # embedding parameters
        self.embed_dim = opts.embed_size
        self.vocab_size = vocab.m_size
        self.pre_embed_path = opts.pre_embed_path
        self.str2idx = vocab.str2idx
        self.embed_uniform_init = opts.embed_uniform_init
        # network parameters
        self.stride = opts.stride
        self.kernel_size = opts.kernel_size
        self.kernel_num = opts.kernel_num
        self.label_num = label_vocab.m_size
        self.embed_dropout = opts.embed_dropout
        self.fc_dropout = opts.fc_dropout

        # embeddings
        self.embeddings = nn.Embedding(self.vocab_size, self.embed_dim)
        if opts.pre_embed_path != '':
            embedding = Embedding.load_predtrained_emb_zero(self.pre_embed_path, self.str2idx)
            self.embeddings.weight.data.copy_(embedding)
        else:
            nn.init.uniform_(self.embeddings.weight.data, -self.embed_uniform_init, self.embed_uniform_init)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.kernel_num, (K, self.embed_dim), stride=self.stride, padding=(K // 2, 0)) for K in
             self.kernel_size])
    
#       torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

        in_fea = len(self.kernel_size) * self.kernel_num

        # dense layer
        # torch.nn.Linear(in_features, out_features, bias=True)
        self.linear1 = nn.Linear(in_fea, in_fea // 2)
        self.linear2 = nn.Linear(in_fea // 2, self.label_num)

        self.embed_dropout = nn.Dropout(self.embed_dropout)
        self.fc_dropout = nn.Dropout(self.fc_dropout)

# define the graph
    def forward(self, input):
        out = self.embeddings(input)
        out = self.embed_dropout(out)
        out = torch.tanh(out)
        l = []
        #  unsqueeze() applied at dim = dim + input.dim() + 1
        out = out.unsqueeze(1)
        # Returns a tensor with all the dimensions of input of size 1 removed.
        for conv in self.convs:
            l.append(torch.tanh(conv(out)).squeeze(3)) # remove the 3rd(4th) dimension
        out = l
        l = []
        for i in out:
            l.append(F.max_pool1d(i, kernel_size=i.size(2)).squeeze(2))
            
        out = torch.cat(l, 1)
        out = self.fc_dropout(out)
        out = self.linear1(F.relu(out))
        out = self.linear2(F.relu(out))
        return out
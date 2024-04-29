
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.init as init




class Multi_CNN(nn.Module):
    
    def __init__(self, opts, vocab, label_vocab):
        super(Multi_CNN, self).__init__()
        
        self.opts = opts
        
        random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        random.seed(opts.seed)
      
        Vocab_size = vocab.m_size
        Vocab_size_mui = opts.embed_num_mui
        embed_dim = opts.embed_dim
        label_num = label_vocab.m_size
        input_num = 2
        kernel_num = opts.kernel_num
        kernel_size = opts.kernel_sizes

        if opts.max_norm is not None:
            print("max_norm = {} ".format(opts.max_norm))
            self.embed_no_static = nn.Embedding(Vocab_size, embed_dim, max_norm=opts.max_norm, scale_grad_by_freq=True, padding_idx=opts.paddingId)
            self.embed_static = nn.Embedding(Vocab_size_mui, embed_dim, max_norm=opts.max_norm, scale_grad_by_freq=True, padding_idx=opts.paddingId_mui)
        else:
            print("max_norm = {} ".format(opts.max_norm))
            self.embed_no_static = nn.Embedding(Vocab_size, embed_dim, scale_grad_by_freq=True, padding_idx=opts.paddingId)
            self.embed_static = nn.Embedding(Vocab_size_mui, embed_dim, scale_grad_by_freq=True, padding_idx=opts.paddingId_mui)
        if opts.word_Embedding:
            self.embed_no_static.weight.data.copy_(opts.pretrained_weight)
            self.embed_static.weight.data.copy_(opts.pretrained_weight_static)
            # whether to fixed the word embedding
            self.embed_no_static.weight.requires_grad = False

        if opts.wide_conv is True:
            print("using wide convolution")
            self.convs1 = [nn.Conv2d(in_channels=input_num, out_channels=kernel_num, kernel_size=(K, embed_dim), stride=(1, 1),
                                     padding=(K//2, 0), bias=True) for K in kernel_size]
        else:
            print("using narrow convolution")
            self.convs1 = [nn.Conv2d(in_channels=input_num, out_channels=kernel_num, kernel_size=(K, embed_dim), bias=True) for K in kernel_size]
        print(self.convs1)

        if opts.init_weight:
            print("Initing W .......")
            for conv in self.convs1:
                init.xavier_normal(conv.weight.data, gain=np.sqrt(opts.init_weight_value))
                init.uniform(conv.bias, 0, 0)
        '''
        self.conv13 = nn.Conv2d(input_num, kernel_num, (3, D))
        self.conv14 = nn.Conv2d(input_num, kernel_num, (4, D))
        self.conv15 = nn.Conv2d(input_num, kernel_num, (5, D))
        '''
        self.dropout = nn.Dropout(opts.dropout)

        # for cnn cuda
        if self.opts.cuda is True:
            for conv in self.convs1:
                conv = conv.cuda()

        in_fea = len(kernel_size) * kernel_num
        self.fc1 = nn.Linear(in_features=in_fea, out_features=in_fea // 2, bias=True)
        self.fc2 = nn.Linear(in_features=in_fea // 2, out_features=label_num, bias=True)

        if opts.batch_normalizations is True:
            print("using batch_normalizations in the model......")
            self.convs1_bn = nn.BatchNorm2d(num_features=kernel_num, momentum=opts.bath_norm_momentum,
                                            affine=opts.batch_norm_affine)
            self.fc1_bn = nn.BatchNorm1d(num_features=in_fea//2, momentum=opts.bath_norm_momentum,
                                         affine=opts.batch_norm_affine)
            self.fc2_bn = nn.BatchNorm1d(num_features=label_num, momentum=opts.bath_norm_momentum,
                                         affine=opts.batch_norm_affine)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) #(N,kernel_num,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x_no_static = self.embed_no_static(x)
        x_static = self.embed_static(x)
        x = torch.stack([x_static, x_no_static], 1)
        x = self.dropout(x)
        if self.opts.batch_normalizations is True:
            x = [F.relu(self.convs1_bn(conv(x))).squeeze(3) for conv in self.convs1] #[(N,kernel_num,W), ...]*len(kernel_size)
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,kernel_num), ...]*len(kernel_size)
        else:
            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,kernel_num,W), ...]*len(kernel_size)
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,kernel_num), ...]*len(kernel_size)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N,len(kernel_size)*kernel_num)
        if self.opts.batch_normalizations is True:
            x = self.fc1(x)
            logit = self.fc2(F.relu(x))
        else:
            x = self.fc1(x)
            logit = self.fc2(F.relu(x))
        return logit
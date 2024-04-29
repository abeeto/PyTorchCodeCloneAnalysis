# Character level text CNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# self defined utils
import utils.Embedding as Embedding


class Char_CNN(nn.Module):

    def __init__(self, opts, vocab, char_vocab, label_vocab):
        super(Char_CNN, self).__init__()

        random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)

        # embedding parameters
        self.embed_dim = opts.embed_size
        self.char_embed_dim = opts.char_embed_size
        self.vocab_size = vocab.m_size
        self.char_num = char_vocab.m_size
        self.pre_embed_path = opts.pre_embed_path
        self.str2idx = vocab.str2idx
        self.char_str2idx = char_vocab.str2idx
        self.embed_uniform_init = opts.embed_uniform_init
        
        # network parameters
        self.stride = opts.stride
        self.kernel_size = opts.kernel_size
        self.kernel_num = opts.kernel_num
        self.label_num = label_vocab.m_size
        self.embed_dropout = opts.embed_dropout
        self.fc_dropout = opts.fc_dropout

        # gpu option
        self.use_cuda = opts.use_cuda

        # embeddings: word level and char level
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embed_dim)
        self.char_embeddings = nn.Embedding(self.char_num, self.char_embed_dim)

        if opts.pre_embed_path != '':
            embedding = Embedding.load_predtrained_emb_zero(self.pre_embed_path, self.str2idx)
            self.word_embeddings.weight.data.copy_(embedding)
        else:
            nn.init.uniform_(self.word_embeddings.weight.data, -self.embed_uniform_init, self.embed_uniform_init)
        
        nn.init.uniform_(self.char_embeddings.weight.data, -self.embed_uniform_init, self.embed_uniform_init)

        word_char_embed_dim = self.embed_dim + len(self.kernel_size) * self.kernel_num

        self.word_char_convs = nn.ModuleList(
            [nn.Conv2d(1, self.kernel_num, (K, word_char_embed_dim), stride=self.stride, padding=(K // 2, 0)) for K in
             self.kernel_size])

        self.char_convs = nn.ModuleList(
            [nn.Conv2d(1, self.kernel_num, (K, self.char_embed_dim), stride=self.stride, padding=(K // 2, 0)) for K in
             self.kernel_size])


        infea = len(self.kernel_size) * self.kernel_num
        self.linear1 = nn.Linear(infea, infea // 2)
        self.linear2 = nn.Linear(infea // 2, self.label_num)

        self.embed_dropout = nn.Dropout(self.embed_dropout)
        self.fc_dropout = nn.Dropout(self.fc_dropout)

    def forward(self, sent, chars_list):
        if self.use_cuda:
            sent = sent.cuda()
        
        #word level embeddings
        sent = self.word_embeddings(sent)
        sent = self.embed_dropout(sent)

        # char through cnn
        char_pooling_list = []
        for sent_chars in chars_list:
            if self.use_cuda:
                sent_chars = sent_chars.cuda()
                
            sent_chars = self.char_embeddings(sent_chars)
            sent_chars = self.embed_dropout(sent_chars)
            char_l = []
            
            sent_chars = sent_chars.unsqueeze(1)
            
            # n-gram at char level
            for conv in self.char_convs:
                char_l.append(torch.tanh(conv(sent_chars)).squeeze(3))

            sent_out = []
            for i in char_l:
                sent_out.append(F.max_pool1d(i, kernel_size=i.size(2)).squeeze(2))

            sent_out = torch.cat(sent_out, 1)

            char_pooling_list.append(sent_out)
            
        # combine/concat char and word    
        word_char_list = []
        word_char = None
        for word, char_pooling in zip(sent, char_pooling_list):
            if word_char is not None:
                word_char = torch.cat((word_char, torch.cat((word, char_pooling), 1).unsqueeze(0)), 0)
            else:
                word_char = torch.cat((word, char_pooling), 1).unsqueeze(0)
            word_char_list.append(word_char)

        # put hybrid features into cov again
        l = []
        word_char = word_char.unsqueeze(1)
        for conv in self.word_char_convs:
            l.append(torch.tanh(conv(word_char)).squeeze(3))
        # print('##', l[0].size())  # torch.Size([32, 100, 2])
        out = []
        for i in l:
            out.append(F.max_pool1d(i, kernel_size=i.size(2)).squeeze(2))
        # print('###', out[0].size())  # torch.Size([32, 100])
        out = torch.cat(out, 1)
        # print('####', out[0].size())  # torch.Size([400])

        out = self.fc_dropout(out)
        out = self.linear1(torch.tanh(out))
        out = self.linear2(torch.tanh(out))

        return out

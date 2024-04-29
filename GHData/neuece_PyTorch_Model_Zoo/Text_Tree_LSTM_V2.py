# Tree-Structured Long Short-Term Memory Networks

# https://www.aclweb.org/anthology/P15-1150


import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.Embedding as Embedding
from torch.autograd import Variable

import random

class ChildSumTreeLSTM(nn.Module):
    def __init__(self, opts, vocab, label_vocab):
        super(ChildSumTreeLSTM, self).__init__()

        random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)
        
        # embedding parameters
        self.embed_dim = opts.embed_size
        self.word_num = vocab.m_size
        self.pre_embed_path = opts.pre_embed_path
        self.str2idx = vocab.str2idx
        self.embed_uniform_init = opts.embed_uniform_init
        self.label_num = label_vocab.m_size
        self.embed_dropout = opts.embed_dropout
        
        # network parameters
        self.fc_dropout = opts.fc_dropout
        self.hidden_size = opts.hidden_size
        self.use_cuda = opts.use_cuda


        self.embeddings = nn.Embedding(self.word_num, self.embed_dim)
        if opts.pre_embed_path != '':
            embedding = Embedding.load_predtrained_emb_zero(self.pre_embed_path, self.str2idx)
            self.embeddings.weight.data.copy_(embedding)

        # build lstm: following the notations as the paper
        
        # input unit
        self.ix = nn.Linear(self.embed_dim, self.hidden_size)
        self.ih = nn.Linear(self.hidden_size, self.hidden_size)
        
        # forget unit
        self.fx = nn.Linear(self.embed_dim, self.hidden_size)
        self.fh = nn.Linear(self.hidden_size, self.hidden_size)

        # output unit
        self.ox = nn.Linear(self.embed_dim, self.hidden_size)
        self.oh = nn.Linear(self.hidden_size, self.hidden_size)
        
        # utility
        self.ux = nn.Linear(self.embed_dim, self.hidden_size)
        self.uh = nn.Linear(self.hidden_size, self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.label_num)

        self.embed_dropout = nn.Dropout(self.embed_dropout)
        self.fc_dropout = nn.Dropout(self.fc_dropout)

        if self.use_cuda:
            self.loss = self.loss.cuda()
            
    

        
    # combine/update status of node j with children node
    def node_forward(self, x, child_c, child_h):
        
        child_h_sum = torch.sum(torch.squeeze(child_h, 1), 0)
        
        # calculate the value of units in LSTM
        i = torch.sigmoid(self.ix(x) + self.ih(child_h_sum)) # equation 3
        o = torch.sigmoid(self.fx(x) + self.fh(child_h_sum)) # equation 5
        u = torch.tanh(self.ux(x) + self.uh(child_h_sum)) # equation 6

        fx = torch.unsqueeze(self.fx(x), 1) # equation 4
        
        f = torch.cat([self.fh(child_i) + fx for child_i in child_h])
        f = torch.sigmoid(f)
        
        fc = torch.squeeze(torch.mul(f, child_c), 1)
        
        c = torch.mul(i, u) + torch.sum(fc, 0) # equation 7
        h = torch.mul(o, torch.tanh(c)) # equation 8

        # print('c.size():', c.size())
        # print('h.size():', h.size())

        return c, h

    # net work structure
    def forward(self, x, tree):
        # print()
        # print(x.size())
        if tree.label is not None:
            x = self.embeddings(x)

        for child in tree.children_list:
            _, _ = self.forward(x, child)
        child_c, child_h = self.get_child_states(tree)

        tree.c, tree.h = self.node_forward(torch.unsqueeze(x[0][tree.index], 0), child_c, child_h)

        output1 = tree.c
        output2 = tree.h

        if tree.label is not None:
            h = self.fc_dropout(tree.h)
            output2 = self.out(h)


        return output1, output2

    def get_child_states(self, tree):


        children_num = len(tree.children_list)

        if children_num == 0:
            c = Variable(torch.zeros((1, 1, self.hidden_size)))
            h = Variable(torch.zeros((1, 1, self.hidden_size)))

        else:
            c = Variable(torch.zeros(children_num, 1, self.hidden_size))
            h = Variable(torch.zeros(children_num, 1, self.hidden_size))
            for idx, child in enumerate(tree.children_list):
                c[idx] = child.c
                h[idx] = child.h

        if self.use_cuda:
            c = c.cuda()
            h = h.cuda()
        return c, h


class BatchChildSumTreeLSTM(nn.Module):
    def __init__(self, opts, vocab, label_vocab):
        super(BatchChildSumTreeLSTM, self).__init__()

        random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)

        self.embed_dim = opts.embed_size
        self.word_num = vocab.m_size
        self.pre_embed_path = opts.pre_embed_path
        self.str2idx = vocab.str2idx
        self.embed_uniform_init = opts.embed_uniform_init
        self.label_num = label_vocab.m_size
        self.embed_dropout = opts.embed_dropout
        self.fc_dropout = opts.fc_dropout
        self.hidden_size = opts.hidden_size
        self.hidden_dropout = opts.hidden_dropout
        self.use_cuda = opts.use_cuda
        self.debug = False

        self.embeddings = nn.Embedding(self.word_num, self.embed_dim)
        if opts.pre_embed_path != '':
            embedding = Embedding.load_predtrained_emb_zero(self.pre_embed_path, self.str2idx)
            self.embeddings.weight.data.copy_(embedding)

        # build lstm
        self.ix = nn.Linear(self.embed_dim, self.hidden_size)
        self.ih = nn.Linear(self.hidden_size, self.hidden_size)

        self.fx = nn.Linear(self.embed_dim, self.hidden_size)
        self.fh = nn.Linear(self.hidden_size, self.hidden_size)

        self.ox = nn.Linear(self.embed_dim, self.hidden_size)
        self.oh = nn.Linear(self.hidden_size, self.hidden_size)

        self.ux = nn.Linear(self.embed_dim, self.hidden_size)
        self.uh = nn.Linear(self.hidden_size, self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.label_num)

        self.hidden_dropout = nn.Dropout(self.hidden_dropout)
        self.embed_dropout = nn.Dropout(self.embed_dropout)
        self.fc_dropout = nn.Dropout(self.fc_dropout)


    def node_forward(self, x, child_c, child_h):
        if self.use_cuda:
            x = x.cuda()
            child_c = child_c.cuda()
            child_h = child_h.cuda()
        
        child_h_sum = torch.sum(child_h, 1)  # torch.Size([4, 100])
        
        i = torch.sigmoid(self.ix(x) + self.ih(child_h_sum))
        o = torch.sigmoid(self.fx(x) + self.fh(child_h_sum))
        u = torch.tanh(self.ux(x) + self.uh(child_h_sum))

        fx = torch.unsqueeze(self.fx(x), 1)  # torch.Size([4, 1, 100])
        

        fx = fx.view(fx.size(0), 1, fx.size(2)).expand(fx.size(0), child_h.size(1), fx.size(2))  # torch.Size([4, 2, 100])
        
        f = self.fh(child_h) + fx  # torch.Size([4, 2, 100])
        

        f = torch.sigmoid(f)
        fc = F.torch.mul(f, child_c)  # torch.Size([4, 2, 100])

        c = torch.mul(i, u) + torch.sum(fc, 1)
        h = torch.mul(o, torch.tanh(c))
        return c, h


    def forward(self, x, bfs_tensor, children_batch_list):
        '''
        :param x: words_id_tensor
        :param bfs_tensor: tensor
        :param children_batch_list: tensor
        :return:
        '''
        x = self.embeddings(x)
        x = self.embed_dropout(x)
        
        batch_size = x.size(0)
        sent_len = x.size()[1]
        all_C = Variable(torch.zeros((batch_size, sent_len, self.hidden_size)))
        all_H = Variable(torch.zeros((batch_size, sent_len, self.hidden_size)))
        if self.use_cuda:
            all_C = all_C.cuda()
            all_H = all_H.cuda()

        h = None
        for index in range(sent_len):
            # get ith embeds
            mask = torch.zeros(x.size())
            # print(mask.size())
            one = torch.ones((1, x.size(2)))
            batch = 0
            for i in torch.transpose(bfs_tensor, 0, 1).data.tolist()[index]:
                mask[batch][i] = one
                batch += 1
            mask = Variable(torch.ByteTensor(mask.data.tolist()))
            if self.use_cuda:
                mask = mask.cuda()
            cur_embeds = torch.masked_select(x, mask)
            cur_embeds = cur_embeds.view(cur_embeds.size(-1) // self.embed_dim, self.embed_dim)

            # select current index from bfs
            mask = []
            mask.extend([0 for _ in range(sent_len)])
            mask[index] = 1
            mask = Variable(torch.ByteTensor(mask))
            if self.use_cuda:
                mask = mask.cuda()
            cur_nodes_list = torch.masked_select(bfs_tensor, mask).data.tolist()

            # select current node's children from children_batch_list
            mask = torch.zeros(batch_size, sent_len, sent_len)
            for i, rel in enumerate(cur_nodes_list):
                mask[i][rel] = torch.ones(1, sent_len)
            mask = Variable(torch.ByteTensor(mask.data.tolist()))
            if self.use_cuda:
                mask = mask.cuda()
            rels = torch.masked_select(children_batch_list, mask).view(batch_size, sent_len)


            rels_sum = torch.sum(rels, 1)
            rels_max = torch.max(rels_sum)

            rel_batch_max = torch.max(rels_max, 0)[0]
            c, h = None, None
            if rel_batch_max.data.tolist() == 0:
                c = Variable(torch.zeros((batch_size, 1, self.hidden_size)))
                h = Variable(torch.zeros((batch_size, 1, self.hidden_size)))
            else:
                pad_c = Variable(torch.zeros(batch_size, rel_batch_max, self.hidden_size))
                pad_h = Variable(torch.zeros(batch_size, rel_batch_max, self.hidden_size))
                rels_broadcast = rels.unsqueeze(1).expand(rels.size(0), self.hidden_size, rels.size(1))
                rels_broadcast = Variable(torch.ByteTensor(rels_broadcast.data.tolist()))
                if self.use_cuda:
                    rels_broadcast = rels_broadcast.cuda()
                    pad_c = pad_c.cuda()
                    pad_h = pad_h.cuda()
                selected_c = torch.masked_select(torch.transpose(all_C, 1, 2), rels_broadcast)
                selected_h = torch.masked_select(torch.transpose(all_H, 1, 2), rels_broadcast)
                selected_c = selected_c.view(selected_c.size(0) // self.hidden_size, self.hidden_size)
                selected_h = selected_h.view(selected_h.size(0) // self.hidden_size, self.hidden_size)
                idx = 0
                for i, batch in enumerate(pad_c):
                    for j in range(rels_sum.data.tolist()[i]):
                        batch[j] = selected_c[idx]
                        idx += 1
                idx = 0
                for i, batch in enumerate(pad_h):
                    for j in range(rels_sum.data.tolist()[i]):
                        batch[j] = selected_h[idx]
                        idx += 1

                c = pad_c
                h = pad_h

            # lstm cell
            c, h = self.node_forward(cur_embeds, c, h)
            h = self.hidden_dropout(h)
            # insert c and h to all_C and all_H
            batch = 0
            for i in cur_nodes_list:
                all_C[batch][i] = c[batch]
                all_H[batch][i] = h[batch]
                batch += 1

        out = torch.transpose(all_H, 1, 2)
        out = torch.tanh(out)
        out = F.max_pool1d(out, out.size(2))
        out = out.squeeze(2)
        out = self.out(out)
        return out
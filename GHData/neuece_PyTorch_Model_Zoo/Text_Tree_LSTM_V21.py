
# Tree-Structured Long Short-Term Memory Networks

# https://www.aclweb.org/anthology/P15-1150


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import utils.Embedding as Embedding
from utils.tree import *
import numpy as np

import random

class ChildSumTreeLSTM_rel(nn.Module):
    def __init__(self, opts, vocab, label_vocab, rel_vocab):
        super(ChildSumTreeLSTM_rel, self).__init__()

        random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)
        # embedding parameters
        self.embed_dim = opts.embed_size
        self.rel_embed_dim = opts.rel_embed_size
        self.word_num = vocab.m_size
        self.pre_embed_path = opts.pre_embed_path
        self.str2idx = vocab.str2idx
        self.embed_uniform_init = opts.embed_uniform_init
        self.label_num = label_vocab.m_size
        self.rel_num = rel_vocab.m_size
        
        # network parameters
        self.dropout = opts.dropout
        self.hidden_size = opts.hidden_size
        self.hidden_num = opts.hidden_num
        self.bidirectional = opts.bidirectional
        self.use_cuda = opts.use_cuda
        self.debug = False

        self.embeddings = nn.Embedding(self.word_num, self.embed_dim)
        self.rel_embeddings = nn.Embedding(self.rel_num, self.rel_embed_dim)
        self.dropout = nn.Dropout(self.dropout)
        if opts.pre_embed_path != '':
            embedding = Embedding.load_predtrained_emb_zero(self.pre_embed_path, self.str2idx)
            self.embeddings.weight.data.copy_(embedding)

        self.dt_tree = DTTreeLSTM(self.embed_dim + self.rel_embed_dim, self.hidden_size, opts.dropout)

        self.linear = nn.Linear(self.hidden_size, self.label_num)

    def forward(self, xs, rels, heads, xlengths):

        emb = self.embeddings(xs)
        rel_emb = self.rel_embeddings(rels)
        outputs = torch.cat([emb, rel_emb], 2)
        outputs = self.dropout(outputs)
        outputs = outputs.transpose(0, 1)

        max_length, batch_size, input_dim = outputs.size()

        trees = []
        indexes = np.zeros((max_length, batch_size), dtype=np.int32)
        for b, head in enumerate(heads):
            root, tree = createTree(head)
            root.traverse()
            for step, index in enumerate(root.order):
                indexes[step, b] = index
            trees.append(tree)

        dt_outputs = self.dt_tree(outputs, indexes, trees, xlengths)

        out = torch.transpose(dt_outputs, 1, 2)
        out = F.max_pool1d(out, out.size(2))
        out = out.squeeze(2)
        out = self.linear(out)
        return out



class DTTreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        """
        """
        super(DTTreeLSTM, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        # linear parameters for transformation from input to hidden state
        # LSTM
        self.i_x = nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)
        self.i_h = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
        self.f_x = nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)
        self.f_h = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
        self.o_x = nn.Linear(in_features=input_size, out_features=hidden_size, bias=False)
        self.o_h = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.u_x = nn.Linear(in_features=input_size, out_features=hidden_size, bias=False)
        self.u_h = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)

    def forward(self, inputs, indexes, trees, lengths):
        """
        :param inputs: batch first
        :param tree:
        :return: output, h_n
        """
        # print('inputs.size():', inputs.size())
        max_length, batch_size, input_dim = inputs.size()
        dt_state_h = []
        dt_state_c = []
        degree = np.zeros((batch_size, max_length), dtype=np.int32)
        last_indexes = np.zeros((batch_size), dtype=np.int32)
        for b, tree in enumerate(trees):
            dt_state_h.append({})
            dt_state_c.append({})
            for index in range(lengths[b]):
                degree[b, index] = tree[index].left_num + tree[index].right_num

        zeros = Var(inputs.data.new(self._hidden_size).fill_(0.))
        # print('zeros.size():', zeros.size())
        for step in range(max_length):
            step_inputs, left_child_hs, right_child_hs, compute_indexes = [], [], [], []
            left_child_cs, right_child_cs = [], []
            for b, tree in enumerate(trees):
                last_index = last_indexes[b]
                for idx in range(last_index, lengths[b]):
                    cur_index = indexes[idx, b]
                    if degree[b, cur_index] > 0:
                        break
                    last_indexes[b] += 1

                    compute_indexes.append((b, cur_index))
                    step_inputs.append(inputs[cur_index, b])
                    if tree[cur_index].left_num == 0:
                        left_child_h = [zeros]
                        left_child_c = [zeros]
                    else:
                        left_child_h = [dt_state_h[b][child.index] for child in tree[cur_index].left_children]
                        left_child_c = [dt_state_c[b][child.index] for child in tree[cur_index].left_children]

                    if tree[cur_index].right_num == 0:
                        right_child_h = [zeros]
                        right_child_c = [zeros]
                    else:
                        right_child_h = [dt_state_h[b][child.index] for child in tree[cur_index].right_children]
                        right_child_c = [dt_state_c[b][child.index] for child in tree[cur_index].right_children]

                    left_child_hs.append(left_child_h)
                    right_child_hs.append(right_child_h)
                    left_child_cs.append(left_child_c)
                    right_child_cs.append(right_child_c)



            if len(compute_indexes) == 0:
                for b, last_index in enumerate(last_indexes):
                    if last_index != lengths[b]:
                        print('bug exists: some nodes are not completed')
                break

            assert len(left_child_hs) == len(right_child_hs)
            assert len(left_child_cs) == len(right_child_cs)
            assert len(left_child_hs) == len(left_child_cs)

            child_hs = []
            child_cs = []
            for i in range(len(left_child_hs)):
                child_h = []
                child_h.extend(left_child_hs[i])
                child_h.extend(right_child_hs[i])
                child_c = []
                child_c.extend(left_child_cs[i])
                child_c.extend(right_child_cs[i])
                child_hs.append(child_h)
                child_cs.append(child_c)
            max_child_num = max([len(child_h) for child_h in child_hs])
            for i in range(len(child_hs)):
                child_hs[i].extend((max_child_num - len(child_hs[i])) * [zeros])
                child_cs[i].extend((max_child_num - len(child_cs[i])) * [zeros])
                child_hs[i] = torch.stack(child_hs[i], 0)
                child_cs[i] = torch.stack(child_cs[i], 0)


            step_inputs = torch.stack(step_inputs, 0)
            child_hs = torch.stack(child_hs, 0)
            child_cs = torch.stack(child_cs, 0)


            h, c = self.node_forward(step_inputs, child_hs, child_cs)
            for idx, (b, cur_index) in enumerate(compute_indexes):
                dt_state_h[b][cur_index] = h[idx]
                dt_state_c[b][cur_index] = c[idx]
                if trees[b][cur_index].parent is not None:
                    parent_index = trees[b][cur_index].parent.index
                    degree[b, parent_index] -= 1
                    if degree[b, parent_index] < 0:
                        print('strange bug')

        outputs, output_t = [], []

        # pads = Var(inputs.data.new(self._hidden_size).fill_(-999999))
        for b in range(batch_size):
            output = [dt_state_h[b][idx] for idx in range(0, lengths[b])] \
                     + [zeros for idx in range(lengths[b], max_length)]
            outputs.append(torch.stack(output, 0))

        return torch.stack(outputs, 0)

    def node_forward(self, input, child_hs, child_cs):

        h_sum = torch.sum(child_hs, 1)

        i = self.i_x(input) + self.i_h(h_sum)
        i = torch.sigmoid(i)

        fx = self.f_x(input)
        fx = fx.unsqueeze(1)
        fx = fx.view(fx.size(0), 1, fx.size(2)).expand(fx.size(0), child_hs.size(1), fx.size(2))
        f = self.f_h(child_hs) + fx
        f = torch.sigmoid(f)

        fc = f * child_cs

        o = self.o_x(input) + self.o_h(h_sum)
        o = torch.sigmoid(o)

        u = self.u_x(input) + self.u_h(h_sum)
        u = torch.tanh(u)

        c = i * u + torch.sum(fc, 1)
        h = o * torch.tanh(c)

        return self.dropout(h), c
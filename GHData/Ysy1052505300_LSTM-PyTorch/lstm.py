from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(LSTM, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.linear_fx = nn.Linear(input_dim, hidden_dim)
        self.linear_fh = nn.Linear(hidden_dim, hidden_dim)
        self.forget = nn.Sigmoid()

        self.linear_ix = nn.Linear(input_dim, hidden_dim)
        self.linear_ih = nn.Linear(hidden_dim, hidden_dim)
        self.input = nn.Sigmoid()
        self.linear_gx = nn.Linear(input_dim, hidden_dim)
        self.linear_gh = nn.Linear(hidden_dim, hidden_dim)
        self.tanh_g = nn.Tanh()

        self.linear_ox = nn.Linear(input_dim, hidden_dim)
        self.linear_oh = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Sigmoid()
        self.tanh_h = nn.Tanh()

        self.linear_ph = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # Implementation here ...
        pre_h = torch.zeros((self.batch_size, self.hidden_dim))
        pre_c = torch.zeros((self.batch_size, self.hidden_dim))
        y = None
        for count in range(self.seq_length):
            xt = torch.reshape(x[count], (self.batch_size, 1))
            ft_x = self.linear_fx(xt)
            ft_h = self.linear_fh(pre_h)
            ft_pre = ft_x + ft_h
            ft = self.forget(ft_pre)

            it_x = self.linear_ix(xt)
            it_h = self.linear_ih(pre_h)
            it_pre = it_x + it_h
            it = self.input(it_pre)
            gt_x = self.linear_gx(xt)
            gt_h = self.linear_gh(pre_h)
            gt_pre = gt_x + gt_h
            gt = self.tanh_g(gt_pre)
            ct = gt.mul(it) + pre_c.mul(ft)

            ot_x = self.linear_ox(xt)
            ot_h = self.linear_oh(pre_h)
            ot_pre = ot_x + ot_h
            ot = self.output(ot_pre)
            ht_pre = self.tanh_h(ct)
            ht = ht_pre.mul(ot)

            pre_c = ct
            pre_h = ht

            if count == self.seq_length - 1:
                pt = self.linear_ph(ht)
                y = self.softmax(pt)
        return y



    # add more methods here if needed
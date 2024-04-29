#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2020 LeonTao
#
# Distributed under terms of the MIT license.

"""

"""

import torch
import torch.nn as nn

from modules.bilstm import BiLSTMEncoder
#from modules.crf import CRF
from modules.crf2 import CRF

class LSTMCRF(nn.Module):
    def __init__(self, src_size, tgt_size,
                 pad_idx, config):
        super(LSTMCRF, self).__init__()

        self.bi_lstm = BiLSTMEncoder(src_size, tgt_size, config)

        #self.crf = CRF(tgt_size, pad_idx)
        self.crf = CRF(tgt_size, pad_idx, False)

    def forward(self, inputs, inputs_length, inputs_mask, tgts):
        '''
        Args:
            inputs: [seq_len, batch_size]
            inputs_length: [batch_size]
            inputs_mask: [seq_len, batch_size]
            tgts: [seq_len, batch_size]
        '''
        emissions = self.bi_lstm(inputs, inputs_length)

        # loss = self.crf(emissions, tgts, inputs_mask)
        loss = self.crf(emissions, tgts, inputs_mask, reduction='mean')

        return loss

    def decode(self, inputs, inputs_length, inputs_mask):
        emissions = self.bi_lstm(inputs, inputs_length)

        return self.crf.decode(emissions, inputs_mask)
        

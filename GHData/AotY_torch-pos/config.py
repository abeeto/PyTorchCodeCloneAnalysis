#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2020 LeonTao
#
# Distributed under terms of the MIT license.

"""

"""


class VocabConfig:
    #def __init__(self):
    UNK = '<unk>'
    PAD = '<pad>'

    SOS = '<sos>'
    EOS = '<eos>'
    max_size = None
    min_freq = 5
    vector = None
    # vector = 'glove.6B.100d '


class ModelConfig:
    # def __init__(self):
    embedding_dim = 300
    hidden_size = 256
    dropout = 0.1

class TrainConfig:
    # def __init__(self):
    model = 'lstm'
    device = 'cpu'
    batch_size = 64
    split_ratio = [0.7, 0.2, 0.1]
    lr = 0.001
    epochs = 50

    seed = 42

    save_dir = './data'



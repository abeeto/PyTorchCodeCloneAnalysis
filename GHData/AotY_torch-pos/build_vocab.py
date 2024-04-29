#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2020 LeonTao
#
# Distributed under terms of the MIT license.

"""
using torchtext vocab
"""

import os
import sys
import argparse
from collections import Counter

import torchtext

UNK = '<unk>'
PAD = '<pad>'

SOS = '<sos>'
EOS = '<eos>'

class Vocab:
    def __int__(self, min_freq=1):
        self.min_freq = min_freq
        self.src_vocab = None
        self.tgt_vocab = None

    def build_vocab(self, input_file):
        src_counter, tgt_counter = self.__build_counter(input_file)
        self.src_vocab = torchtext.vocab.Vocab(src_counter,
                                               min_freq=self.min_freq,
                                               specials=[PAD, UNK])
        self.tgt_vocab = torchtext.vocab.Vocab(tgt_counter,
                                               min_freq=self.min_freq,
                                               specials=[PAD, UNK, SOS, EOS])

    def __build_counter(self, input_file):
        src_counter = Counter()
        tgt_counter = Counter()
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # char, processed tag
                c, t = line.split()
                src_counter.update([c])
                tgt_counter.update([t])

    def save(self, save_dir):
        pickle.dumps(self.src_vocab, open(os.path.join(save_dir, 'src_vocab.pkl'), 'wb'))
        pickle.dumps(self.tgt_vocab, open(os.path.join(save_dir, 'tgt_vocab.pkl'), 'wb'))

    def load(self, save_dir):
        self.src_vocab = pickle.loads(open(os.path.join(save_dir, 'src_vocab.pkl'), 'rb'))
        self.tgt_vocab = pickle.loads(open(os.path.join(save_dir, 'tgt_vocab.pkl'), 'rb'))

if __name__ == '__name__':
    parser = argparse.ArgumentParser(sys.argv[0], description='build vocab by passing pos train file. (each line contains a char and a processed tag)')
    parser.add_argument('input_file', help='processed input file, each line has two columns, one is char and another is processed tag')
    parser.add_argument('save_dir', help='the dir path to save vocab.')
    parser.add_argument('min_freq', default=3, help='the minimum frequency needed to include a token in the vocab.')

    args = parser.parse_args()

    vocab = Vocab(arg.min_freq)
    vocab.build_vocab(args.input_file)
    vocab.save(args.save_dir)


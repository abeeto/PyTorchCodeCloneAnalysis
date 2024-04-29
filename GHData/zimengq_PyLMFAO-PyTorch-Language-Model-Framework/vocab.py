#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>
# Licensed under the Apache License v2.0 - http://www.apache.org/licenses/

"""
Text vocabulary classes definition
"""

from __future__ import unicode_literals

import io
import logging
import torch

from collections import Counter, defaultdict
from functools import reduce

logger = logging.getLogger(__name__)
torch.manual_seed(10707)


def _default_unk_index():
    return 0


def _default_s_index():
    return 1


def _rand_int(dim):
    num = torch.randn(dim)
    return num * torch.rsqrt(num)


def _uni_int(dim):
    num = torch.rand(dim)
    return num * torch.rsqrt(num)


def _zero_int(dim):
    return torch.zeros((dim))


class Vocab(object):
    """
    Defines a vocabulary object that will be used to numericalize a field.
    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """
    def __init__(self, words=None, vectors=None, unk_init=None):
        """
        Create a Vocab object from a collections.Counter.
        Arguments:
            words: Corpus for building vocabulary.
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors
        """
        self.itos = ["<pad>", "<unk>"]
        self.stoi = defaultdict(lambda: len(self.stoi))
        self.stoi["<pad>"] = 0
        self.stoi["<unk>"] = 1

        self.unk_init = _uni_int if unk_init is None else unk_init
        self.pre_trained = vectors
        self.vectors = Vectors(self.unk_init)
        self.vectors['<pad>'] = _zero_int(self.vectors.dim)
        self.vectors['<unk>'] = _rand_int(self.vectors.dim)

        if words is not None:
            self.build(words)

    def build(self, words):
        if isinstance(words, list):
            counter = Counter(reduce(lambda x, y: x + y, words))
        elif isinstance(words, str):
            counter = Counter(words.split())

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

        if self.pre_trained is not None:
            self.load_vectors(self.pre_trained)

    def __eq__(self, other):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.stoi:
                return self.stoi[item]
            else:
                return self.stoi['<unk>']
        elif isinstance(item, int):
            return self.itos[item]
        else:
            raise TypeError("index type could only be either str or int.")

    def extend(self, words, sort=False):
        if isinstance(words, str):
            words = words.split()
        if sort:
            words = sorted(words)
        for w in words:
            if w not in self.stoi.keys():
                self.itos.append(w)
                self.stoi[w] = self.stoi[w]

    def load_vectors(self, vectors=None):
        """
        Arguments:
            vectors: one of or a list containing instantiations of the
                GloVe, CharNGram, or Vectors classes. Alternatively, one
                of or a list of available pretrained vectors:
        """
        if vectors is not None:
            fin = io.open(vectors, 'r', encoding='utf-8', newline='\n', errors='ignore')
        else:
            fin = io.open(self.pre_trained, 'r', encoding='utf-8', newline='\n', errors='ignore')
        for line in fin:
            if len(line.rstrip().split(' ')) == 2:
                print("skip %s, probably file head" % line.rstrip())
                continue
            tokens = line.rstrip().split(' ')
            self.vectors[tokens[0]] = torch.FloatTensor(list(map(float, tokens[1:])))
            self.itos.append(tokens[0])
        self.stoi.update({tok: i for i, tok in enumerate(self.itos) if tok not in ['<unk>', '<pad>']})

    def vector_tensor(self):
        return torch.stack([self.vectors[s] for s in self.itos])


class Vectors(dict):
    """
    Define Vectors class
    """
    def __init__(self, unk_init, dim=300):
        super(dict).__init__()
        self.unk_init = unk_init
        self.dim = dim

    def __getitem__(self, item):
        if item not in self.keys():
            return self.unk_init(self.dim)
        else:
            return dict.__getitem__(self, item)



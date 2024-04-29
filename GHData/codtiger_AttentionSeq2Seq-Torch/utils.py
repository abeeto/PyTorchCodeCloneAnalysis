from __future__ import unicode_literals, print_function, division
import unicodedata
import string
import re
import random
import time
import math
from typing import Tuple

import torch
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class Progress:

    def __init__(self, epochs, steps):
        self.start = time.time()
        self.epochs = epochs
        self.num_steps = steps
        self.elapsed = 0
        self.remained = 0

    def _calculate_progress(self, cur_ep, cur_step):
        return ((( cur_ep - 1) * self.num_steps + cur_step + 1)* 100) / (self.epochs * self.num_steps)

    def update(self, cur_ep, cur_step):
        now = time.time()
        self.elapsed = now - self.start
        estimated_total = self.elapsed / percent
        self.remained = estimated_total - self.elapsed
        self.progress = self._calculate_progress(cur_ep, cur_step)

    def __str__(self):
        return f'{self.to_min(self.elapsed)}, left:{self.to_min(self.remained)}, progress: {self.progress:.3f}%'

    @staticmethod
    def to_min(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_langs(lang1, lang2, filename, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(filename, encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')[:2]] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)


def tensors_from_pair(pair, input_lang, output_lang):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def filter_pairs(pairs, max_length=10, eng_prefixes: Tuple[str]=None):

    def filter_pair(p):
        include = len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length
        if eng_prefixes and include:
            return p[1].startswith(eng_prefixes)
        return include

    return [pair for pair in pairs if filter_pair(pair)]

def loss_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def plot_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
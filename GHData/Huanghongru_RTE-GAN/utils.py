import re
import string
import time
import torch
import random
import unicodedata
import numpy as np
import json_lines as jsonl

SOS_token = 0   # start of the sentence
EOS_token = 1   # end of the sentence
PAD_token = 2   # meaningless padding token
MAX_LENGTH = 48 # the max length of a sentence
w2v_dim = 300   # dimension of word2ec mdoel
w2v = 'glove.6B.%sd.txt' % w2v_dim   # the word2vec model file

class Lang:
    """
    Build the vocabulary for the corpus
    """
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<SOS>", 1: "<EOS>", 2: "<PAD>"}
        self.n_words = 3

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


class phPair:
    """
    A class for premise-hypothesis pair.
    """
    def __init__(self, p, h, l, max_len=MAX_LENGTH, padding=False):
        self.l_dict = {'entailment':0, 'contradiction': 1, 'neutral': 2}
        self.max_len = max_len
        self.label = self.l_dict[l]

        p_ = p.split()
        h_ = h.split()

        # do padding
        if padding:
            p_pad = ["<PAD>" for i in range(self.max_len-len(p_))]
            h_pad = ["<PAD>" for i in range(self.max_len-len(h_))]

            p_ = p_ + p_pad
            h_ = h_ + h_pad
        
        self.premise = ' '.join(p_)
        self.hypothesis = ' '.join(h_)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def getLabel(anno_labels):
    cnt = {'neutral': 0, 'contradiction': 0, 'entailment': 0}
    res, cur_max = '', 0
    for label in anno_labels:
        if label:
            cnt[label] += 1
            if cnt[label] > cur_max:
                res = label
                cur_max = cnt[label]
    return res

def load_pretrained_embedding(lang, w2v_file):
    data, w2v = {}, []

    # add two special token EOS, SOS and PAD
    w2v.append(np.random.rand(w2v_dim).tolist())
    w2v.append(np.random.rand(w2v_dim).tolist())
    
    w2v.append(np.zeros(w2v_dim).tolist())
    unk = np.random.rand(w2v_dim).tolist()
    lines = open(w2v_file, 'r').readlines()
    for line in lines:
        line = line.split()
        data[line[0]] = [eval(d) for d in line[1:]]

    for i in range(3, lang.n_words):
        if i % 1000 == 0:
            print '%d / %d' % (i+3, len(lines))
        if lang.index2word[i] in data.keys():
            w2v.append(data[lang.index2word[i]])
        else:   # unknown words
            w2v.append(unk)
    return torch.tensor(w2v)

def readData(file_name, word2vec=w2v, load_w2v=True):
    """
    Read the data. Filter out those neutral data.
    For datum that has multiple labels, select the most one as its label.

    Input: (str) file_name
    Output: (list) premise-hypothesis pairs
    """
    print "Reading data file %s..." % file_name

    ph_pairs = []
    label_cnt = {'entailment': 0, 'contradiction': 0, 'neutral': 0}
    corpus_dict = Lang('en')
    with open(file_name, 'rb') as f:
        for item in jsonl.reader(f):
            p = normalizeString(item['sentence1'])
            h = normalizeString(item['sentence2'])
            l = getLabel(item['annotator_labels'])
            label_cnt[l] += 1
            datum = phPair(p, h, l)
            if datum.label != 'neutral':
                ph_pairs.append(datum)
                corpus_dict.addSentence(p)
                corpus_dict.addSentence(h)

    print "Loading dataset completed !"
    print "Loading word2vec model..."

    glove = np.zeros((5,5))
    if load_w2v:
        glove = load_pretrained_embedding(corpus_dict, word2vec)
        print "Loading word2vec done!"

    print "Courpus used %d words" % corpus_dict.n_words
    print "Data distributions: %s" % label_cnt
    return ph_pairs, corpus_dict, glove


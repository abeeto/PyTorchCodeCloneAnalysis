import collections
import re
import random

import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from d2l import torch as d2l
import jieba

def read_novel(filepath='/data/datasets/d2l_data/cover_sky.txt'):
    with open(filepath, 'r', encoding='gbk') as f:
        lines = f.readlines()
        # print('len of lines:', len(lines))
        # 简单的把空行、空格都去掉
        return [re.sub('\s', ' ', line) for line in lines]
    
def tokenize(lines, token='char', language='chinese'):
    # print('token = ', token)
    if token == 'word':
        if language == 'chinese':
            # print(jieba.lcut(lines[0], cut_all=False))
            return [jieba.lcut(line, cut_all=False) for line in lines]
        if language == 'english':
            return [line.split() for line in lines]
    if token == 'char':
        return [list(line) for line in lines]
    else:
        print('error token：', token)
        
def corpus_counter(tokens):
    if len(tokens) == 0 or isinstance(tokens, list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
            
        counter = corpus_counter(tokens)
        self.token_freq = sorted(counter.items(), key=lambda x:x[1], reverse=True)
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        self.idx_to_token, self.token_to_idx = [], dict()
        uniq_tokens += [
            token for token, freq in self.token_freq
            if freq > min_freq and token not in uniq_tokens
        ]
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1
            
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (tuple, list)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_token(self, indices):
        if not isinstance(indices, (tuple, list)):
            return self.idx_to_token[indices]
        return [self.to_token(index) for index in indices]


class MyDataSet(Dataset):
    def __init__(self, corpus, num_steps):
        self.corpus = corpus
        self.num_steps = num_steps
        # 要把corpus变成若干个长度为num_steps的子序列，然后在getitem中返回
        self.corpus = self.corpus[random.randint(0, num_steps - 1):]
        num_subseq = (len(self.corpus) - 1) // num_steps
        self.initial_indices = list(range(0, num_subseq * num_steps, num_steps))
            
    def __len__(self):
        return len(self.initial_indices)
    # dataloader会返回一个batch_size个__getitem__的数据
    # 正常RNN的输入是[batch_size, num_steps, vocab_size]
    # 在这里需要考虑num_step
    def __getitem__(self, indices):
        def data(pos):
            return torch.tensor(self.corpus[pos:pos + self.num_steps])
        return data(self.initial_indices[indices]), data(self.initial_indices[indices] + 1)


def load_novel(token, language, max_tokens=-1):
    lines = read_novel()
    tokens = tokenize(lines, token, language)
    vocab = Vocab(tokens, 0)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab
    
def load_data_novel(batch_size, time_steps, token, language, max_tokens):
    corpus, vocab = load_novel(token, language, max_tokens)
    dataset = MyDataSet(corpus, time_steps)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        # shuffle=True,
                        num_workers=8,
                        pin_memory=True,
                        sampler=train_sampler
                        )
    return dataloader, vocab

def trans_dim(state):
    if isinstance(state, (tuple, list)):
        return [s.permute([1, 0, 2]).contiguous() for s in state]
    else:
        return state.permute([1, 0, 2]).contiguous()
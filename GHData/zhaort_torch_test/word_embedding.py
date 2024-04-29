# coding=gbk

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from collections import Counter
import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

USE_CUDA = torch.cuda.is_available()

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if USE_CUDA:
    torch.cuda.manual_seed(1)

CONTEXT_WINDOW = 3
NUM_NEGATIVE = 100
EPOCHS = 2
MAX_VOCAB_SIZE = 30000
BATCH_SIZE = 128
LEARNING_RATE = 0.02
EMBEDDING_SIZE = 100


FILE_PATH = r"D:\datasets\word_embedding\text8.train.txt"
with open(FILE_PATH, "r") as f:
    text = f.read()

text = text.split()
vocab = Counter(text).most_common(MAX_VOCAB_SIZE - 1)
vocab = dict(vocab)
vocab['<unk>'] = len(text) - np.sum(list(vocab.values()))

idx_to_word = [word for word in vocab.keys()]
word_to_idx = {word: i for i, word in enumerate(idx_to_word)}

word_counts = np.array([count for count in vocab.values()])
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3/4)
MAX_VOCAB_SIZE = len(idx_to_word)


class WordEmbeddingDataset(Data.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        super(WordEmbeddingDataset, self).__init__()
        # text中各个单词编码成word_to_idx中的索引
        self.text_encoded = [word_to_idx.get(word, word_to_idx["<unk>"]) for word in text]
        self.text_encoded = torch.LongTensor(self.text_encoded)
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)

    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self, idx):
        center_word = self.text_encoded[idx]
        # 周围单词的索引
        pos_indices = list(range(idx - CONTEXT_WINDOW, idx)) + list(range(idx+1, idx+CONTEXT_WINDOW+1))
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices]
        # negative sampling
        neg_words = torch.multinomial(self.word_freqs, NUM_NEGATIVE * pos_words.shape[0], True)
        return center_word, pos_words, neg_words


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)

    def forward(self, input_labels, pos_labels, neg_labels):
        input_embedding = self.in_embed(input_labels)
        pos_embedding = self.in_embed(pos_labels)
        neg_embedding = self.in_embed(neg_labels)

        input_embedding = input_embedding.unsqueeze(2)
        pos_dot = torch.bmm(pos_embedding, input_embedding).squeeze(2)
        neg_dot = torch.bmm(neg_embedding, -input_embedding).squeeze(2)

        log_pos = F.logsigmoid(pos_dot).sum(1)
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = log_pos + log_neg

        return -loss

    def input_embedding(self):
        return self.in_embed.weight.data.cpu().numpy()


dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)
data_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

model = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
if USE_CUDA:
    model = model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
for epoch in range(EPOCHS):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(data_loader):
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()
        if USE_CUDA:
            input_labels = input_labels.cuda()
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()

        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("epoch", epoch, "iteration", i, loss.item())


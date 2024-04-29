# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import util

char2id = util.get_char2id()

class Encoder(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim):
    super(Encoder, self).__init__()
    self.hidden_dim = hidden_dim
    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=char2id[' '])
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

  def forward(self, sequence):
    embedding = self.word_embeddings(sequence)
    # Many to oneなので、第２戻り値を使う
    _output, state = self.lstm(embedding)
    # state = (h, c)
    return state

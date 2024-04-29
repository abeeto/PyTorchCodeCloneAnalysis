# -*- coding: utf-8 -*-

import torch.nn as nn
import util

char2id = util.get_char2id()

class Decoder(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim):
    super(Decoder, self).__init__()
    self.hidden_dim = hidden_dim
    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=char2id[' '])
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
    # LSTMの128次元の隠れ層を13次元に変換する全結合層
    self.hidden2linear = nn.Linear(hidden_dim, vocab_size)

  def forward(self, sequence, encoder_state):
    embedding = self.word_embeddings(sequence)
    # Many to Manyなので、第１戻り値を使う。
    # 第２戻り値は推論時に次の文字を生成するときに使います。
    output, state = self.lstm(embedding, encoder_state)
    # state = (h, c)
    output = self.hidden2linear(output)
    return output, state

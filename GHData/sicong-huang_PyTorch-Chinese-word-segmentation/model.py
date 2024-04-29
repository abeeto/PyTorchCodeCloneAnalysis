import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, batch_size, vocab_size, tagset_size, layer, bidirect, dropout):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer = layer
        self.batch_size = batch_size
        self.direct = 2 if bidirect else 1
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=layer, bidirectional=bidirect, dropout=dropout)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.layer * self.direct, self.batch_size, self.hidden_dim),
                torch.zeros(self.layer * self.direct, self.batch_size, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.softmax(tag_space, dim=2)
        return tag_scores

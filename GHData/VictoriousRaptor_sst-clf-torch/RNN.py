import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class myRNN(nn.Module):
    def __init__(self, config):
        super(myRNN, self).__init__()

        self.hidden_size = 30  # Dimension of hidden state
        self.embedding_length = config.wordvec_dim
        self.num_layers = 2  # Stack layers
        self.embedding_length = config.wordvec_dim  
        self.word_embeddings = nn.Embedding(
            config.weight.shape[0], self.embedding_length)  # Embedding layer
        self.word_embeddings = self.word_embeddings.from_pretrained(
            config.weight, freeze=False)  # Load pretrianed word embedding, and fine-tuing
        self.recurrent = nn.RNN(self.embedding_length, self.hidden_size,
                                num_layers=self.num_layers, bidirectional=True, batch_first=True)  # Recurrent layer
        self.fc = nn.Linear(2*self.num_layers *
                            self.hidden_size, config.label_num)  # FC layer

    def forward(self, input_sentences):
        
        x = self.word_embeddings(input_sentences)  # (batch_size, batch_dim, embedding_length)
        # print(x.size())

        output, h_n = self.recurrent(x)  # 特征，隐状态
        # print(h_n.size())
        # h_n.size() = (2*self.num_layers, batch_size, hidden_size), 2 for bidirectional

        # (batch_size, 2*self.num_layers, hidden_size)
        h_n = h_n.permute(1, 0, 2)

        h_n = h_n.contiguous().view(h_n.size()[0], h_n.size()[1]*h_n.size()[2])  # (batch_size, 4*hidden_size)
        # print(h_n.size())

        logits = self.fc(h_n)  # (batch_size, label_num)

        return F.softmax(logits, dim=1)


class LSTMClassifier(nn.Module):
    def __init__(self, config):
        super(LSTMClassifier, self).__init__()

        self.label_num = config.label_num
        self.hidden_size = 150
        self.embedding_length = config.wordvec_dim 
        self.word_embeddings = nn.Embedding(
            config.weight.shape[0], self.embedding_length)  # Embedding layer
        self.word_embeddings = self.word_embeddings.from_pretrained(
            config.weight, freeze=False)  # Load pretrianed word embedding, and fine-tuing

        self.lstm = nn.LSTM(self.embedding_length,
                            self.hidden_size, batch_first=True)  # lstm
        self.fc = nn.Linear(self.hidden_size, self.label_num)

    def forward(self, input_sentence):
        x = self.word_embeddings(input_sentence)  # (batch_size, batch_dim, embedding_length)

        output, (final_hidden_state, final_cell_state) = self.lstm(x)
        
        logits = self.fc(final_hidden_state[-1])  # final_hidden_state.size() = (1, batch_size, hidden_size) & logits.size() = (batch_size, label_num)

        return F.softmax(logits, dim=1)

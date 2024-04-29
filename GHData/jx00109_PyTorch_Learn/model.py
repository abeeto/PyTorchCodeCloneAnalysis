# -*- encoding:utf8 -*-
import torch
import torch.nn as nn



class myLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(myLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        # self.hidden2tag = nn.Linear(in_features=hidden_dim, out_features=target_size)

        self.init_emb()
        # self.hidden = self.init_hidden()

    # 在0附近初始化词向量矩阵
    def init_emb(self):
        initrange = 0.5 / self.embedding_dim
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)

    # 初始化lstm的h和c，没有特殊需求可以不写，默认以0初始化
    '''
    def init_hidden(self):
        pass
    '''

    # 在这样输入补长后的句子以及对应的mask [batch_size, time_step, embedding]

    def forward(self, sentence, mask, time_step, batch_size):
        #输出初始时的词向量矩阵
        print self.word_embeddings.weight
        embeds = self.word_embeddings(sentence)
        # lstm的输入形状 [batch_size, time_step, embedding_dim]
        # lstm_h是LSTM最后一层time_step个h的集合，形状为 [batch_size, time_step, hidden_dim]
        # None表示hidden state会用全0的state
        lstm_h, _ = self.lstm(embeds, None)

        # 输出经过lstm处理后的batch
        print lstm_h

        # 由于lstm是三维张量，mask是二维tensor，我们需要将mask拓展成三维，在进行相乘，从而得到每个输入句子真正的最后状态h
        final_h = torch.mul(lstm_h, mask[:, :, None])

        # 由于其他位置上都是0，直接求和在第二维度上求和就可以得到最后状态 [batch, hidden_dim]
        return torch.sum(final_h, 1)

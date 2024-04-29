import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
# torch.manual_seed(233)
# random.seed(233)
# torch.backends.cudnn.enabled = False
import torch.nn.init as init
import Load_embedding

class BiLSTM(nn.Module):
    def __init__(self, args):
        super(BiLSTM, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.dropout = nn.Dropout(args.dropout)
        self.dropout_embed = nn.Dropout(args.dropout_embed)

        self.word_embeddings = nn.Embedding(args.embed_num, args.embedding_dim)
        # self.word_embedding = Load_embedding.Embedding(args.embed_num, args.embedding_dim)
        self.lstm = nn.LSTM(args.embedding_dim, args.hidden_dim, bidirectional=True, dropout=args.dropout_model)

        self.hidden2label1 = nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        self.hidden2label2 = nn.Linear(args.hidden_dim, args.class_num)
        self.hidden = self.init_hidden(args.batch_size)
        self.use_cuda = args.use_cuda

        self.use_pretrained_emb = args.use_pretrained_emb

        if self.use_pretrained_emb:
            # print(args.pretrained_weight)
            # pretrained_weight = np.array(args.pretrained_weight)
            # pretrained_weight = torch.FloatTensor(args.pretrained_weight)
            pretrained_weight = np.array(args.pretrained_weight)
            # print(pretrained_weight)
            # self.word_embedding_static = Load_embedding.ConstEmbedding(pretrained_weight)
            # print(pretrained_weight.shape)
            self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_weight))


    def init_hidden(self, batch_size):
        # the first is the hidden h
        # the second is the cell  c
        if self.args.use_cuda:
            return (Variable(torch.zeros(2, batch_size, self.hidden_dim)).cuda(),
                    Variable(torch.zeros(2, batch_size, self.hidden_dim)).cuda())
        else:
            return (Variable(torch.zeros(2, batch_size, self.hidden_dim)),
                Variable(torch.zeros(2, batch_size, self.hidden_dim)))

    def forward(self, sentence):
        # print(sentence)                                     # [torch.LongTensor of size 42x64]
        # x = self.word_embedding.forward(sentence)
        # self.word_embedding_static.cuda()
        # x = self.word_embedding_static.forward(sentence)
        if self.args.use_cuda:
            sentence = sentence.cuda()
        x = self.word_embeddings(sentence)

        x = self.dropout_embed(x)
        # print(embeds.size())                                # torch.Size([42, 64, 100])
        # x = embeds.view(len(sentence), self.batch_size, -1)
        # print(x.size())                                     # torch.Size([42, 64, 100])
        lstm_out, self.hidden = self.lstm(x, self.hidden)   # lstm_out 10*5*50 hidden 1*5*50 *2
        # print(lstm_out.size())                              # torch.Size([32, 16, 600])
        # print(lstm_out)
        # lstm_out = [F.max_pool1d(i, len(lstm_out)).unsqueeze(2) for i in lstm_out]
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)

        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2))
        # print(lstm_out.size())
        lstm_out = lstm_out.squeeze(2)
        # y = self.hidden2label(lstm_out)

        #lstm_out = torch.cat(lstm_out, 1)
        # lstm_out = self.dropout(lstm_out)
        # lstm_out = lstm_out.view(len(sentence), -1)
        y = self.hidden2label1(F.tanh(lstm_out))
        y = self.hidden2label2(F.tanh(y))
        # log_probs = F.log_softmax(y)

        log_probs = y
        return log_probs

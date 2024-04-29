import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)


class Linear(Bottle, nn.Linear):
    pass


class FeedForwardLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation=nn.SELU(), dp_ratio=0.2):
        super(FeedForwardLayer, self).__init__()
        self.name = 'FeedForwardLayer'

        self.Layer1 = Linear(in_dim, hidden_dim)
        self.Layer2 = Linear(hidden_dim, out_dim)

        self.activation = activation
        self.dropout = nn.Dropout(p=dp_ratio)

    def forward(self, x):
        x = self.Layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.Layer2(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class DecomposableModel(nn.Module):
    def __init__(self, config):
        super(DecomposableModel, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.projection = Linear(config.d_embed, config.d_proj)
        self.dropout = nn.Dropout(p=config.dp_ratio)
        self.activation = nn.SELU()

        self.F = FeedForwardLayer(config.d_embed, config.d_hidden, config.d_F, self.activation, config.dp_ratio)
        self.G = FeedForwardLayer(2 * config.d_embed, config.d_hidden, config.d_G, self.activation, config.dp_ratio)
        self.H = FeedForwardLayer(2 * config.d_G, config.d_hidden, config.d_out, self.activation, config.dp_ratio)

    def forward(self, batch):
        prem_embed = self.embed(batch.premise.transpose(0, 1))
        hypo_embed = self.embed(batch.hypothesis.transpose(0, 1))

        if self.config.fix_emb:
            prem_embed = Variable(prem_embed.data)
            hypo_embed = Variable(hypo_embed.data)

        e = torch.bmm(self.F(prem_embed), self.F(hypo_embed).transpose(1, 2))
        e_ = F.softmax(e)
        e_t = F.softmax(e.transpose(1, 2))

        beta = torch.bmm(e_, hypo_embed)
        alpha = torch.bmm(e_t, prem_embed)

        v1 = self.G(torch.cat((prem_embed, beta), 2)).sum(1)
        v2 = self.G(torch.cat((hypo_embed, alpha), 2)).sum(1)

        v = F.softmax(self.H(self.dropout(torch.cat((v1, v2), 1))).squeeze())
        return v


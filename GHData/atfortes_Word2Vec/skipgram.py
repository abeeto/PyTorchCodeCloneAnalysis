import torch
import torch.nn as nn
from torch.autograd import Variable


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.center_embeds = nn.Embedding(vocab_size, embed_size, sparse=True)
        self.context_embeds = nn.Embedding(vocab_size, embed_size, sparse=True)
        self.init_weights()

    def init_weights(self):
        irange = 0.5 / self.embed_size
        nn.init.uniform_(self.center_embeds.weight.data, -irange, irange)
        nn.init.constant_(self.context_embeds.weight.data, 0)

    def forward(self, pos_center, pos_context, pos_neg_samples):
        center_emb = self.center_embeds(Variable(pos_center))
        context_emb = self.context_embeds(Variable(pos_context))
        neg_emb = self.context_embeds(Variable(pos_neg_samples))

        scores = torch.mul(center_emb, context_emb)
        scores = torch.sum(scores, dim=1)
        scores = nn.functional.logsigmoid(scores)

        neg_scores = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze()
        neg_scores = torch.sum(neg_scores, dim=1)
        neg_scores = nn.functional.logsigmoid(-neg_scores)

        return -(scores + neg_scores).sum() / pos_center.shape[0]

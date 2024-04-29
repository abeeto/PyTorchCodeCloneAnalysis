
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx=0):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)

    def forward(self, input):
        return self.embedding(input)

class ConstEmbedding(nn.Module):
    def __init__(self, pretrained_embedding, padding_idx=0):
        super(ConstEmbedding, self).__init__()
        self.vocab_size = pretrained_embedding.size(0)
        self.embedding_size = pretrained_embedding.size(1)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, sparse=True)
        self.embedding.weight = nn.Parameter(pretrained_embedding, requires_grad=False)

    def cuda(self, device_id=None):
        """
           The weights should be always on cpu
       """
        return self._apply(lambda t: t.cpu())

    def forward(self, input):
        """
           return cpu tensor
       """
        return self.embedding(input)










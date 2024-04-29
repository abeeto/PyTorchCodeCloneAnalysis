__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import torch.nn.functional as F
import torch.nn as nn
import torch


class ModelConfig(object):
    embedding_dim = 128
    vocab_size = 10002
    num_filters = 100
    kernel_sizes = [3, 4, 5]
    dropout_prob = 0.5
    num_classes = 2
    max_len = 50


class TextCNN(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super(TextCNN, self).__init__()
        self.config = config
        self.v = config.vocab_size
        self.e = config.embedding_dim
        self.nf = config.num_filters
        self.ks = config.kernel_sizes
        self.nc = config.num_classes
        self.do = config.dropout_prob

        self.embedding = nn.Embedding(self.v, self.e)
        self.convs = nn.ModuleList([nn.Conv1d(self.e, self.nf, k) for k in self.ks])
        self.dropout = nn.Dropout(self.do)
        self.fc1 = nn.Linear(len(self.ks) * self.nf, self.nc)

    def forward(self, inputs):
        """

        :param inputs: [bs x seq]
        :return:
        """
        # [bs x embed_dim x seq_len]
        e = torch.einsum('ijk->ikj', self.embedding(inputs))
        r = [F.relu(F.max_pool1d(conv(e), self.config.max_len - nf + 1)).squeeze(dim=2)
             for conv, nf in zip(self.convs, self.ks)]

        # [bs x (embed * len(kernel_sizes))]
        r = torch.cat(r, dim=1)

        # [bs x num_class]
        l = self.fc1(self.dropout(r))

        return l


if __name__ == '__main__':
    inputs = torch.rand(32, 50).long()
    print(TextCNN(ModelConfig())(inputs).shape)

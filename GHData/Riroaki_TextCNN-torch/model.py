import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """CNN model."""

    def __init__(self, vocab_size: int, embedding_dim: int, n_filters: int, filter_sizes: list,
                 output_dim: int, dropout: float, pad_idx):
        """Initialization of CNN model.

        :param vocab_size: int, size of vocabulary.
        :param embedding_dim: int, dimension of word e
        :param n_filters: int, count of filters
        :param filter_sizes: list, dimensions of each filter
        :param output_dim: int, output size
        :param dropout: float, rate of drop out
        :param pad_idx: int, padding index
        """

        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.conv_0 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[0], embedding_dim))
        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[1], embedding_dim))
        self.conv_2 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[2], embedding_dim))
        # Full connected layer.
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        # Drop out to reduce over-fitting.
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        """Defines the forward propagation process.

        :param text: input data
        :return: output of model,the predictions
        """

        # text = [sent len, batch size]
        text = text.permute(1, 0)
        # text = [batch size, sent len]
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]
        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)

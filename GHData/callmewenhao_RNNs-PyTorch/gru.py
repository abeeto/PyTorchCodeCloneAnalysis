# 简单使用pytorch的RNN模块
import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, seq_len, embedding_dim=300, hidden_size=100, num_layers=2):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(seq_len, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers)  # 输入维度embedding_dim，输出维度hidden_size，其中全连接层数num_layers
        self.fc = nn.Linear(in_features=seq_len * hidden_size, out_features=10)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # input x shape is: BatchSize, Seq_len
        x = self.embedding(x)  # BatchSize, Seq_len, embedding_dim
        x = torch.permute(x, (1, 0, 2))  # RNN input shape is: Seq_len, BatchSize, embedding_dim

        # init h0, shape is num_layers, BatchSize, hidden_size
        _, B, E = x.shape
        h0 = torch.randn(self.num_layers, B, self.hidden_size)

        # calculate rnn, get: output, hn
        x, hn = self.rnn(x, h0)

        # 送给全连接层
        x = torch.permute(x, (1, 0, 2))  # get original shape: BatchSize, Seq_len, embedding_dim
        x = torch.flatten(x, 1)  # shape is: BatchSize, Seq_len*embedding_dim
        x = self.fc(x)  # get shape: BatchSize, 10
        return self.tanh(x)


def main():
    # Hyper parameters
    batch_size = 2
    seq_len = 20

    # data
    x = torch.rand(size=(batch_size, seq_len)) * 20  # embedding层的输入的值有一个要求：大小不可以超过num_embedding
    x = x.long()
    print(f"input shape is {x.shape}")

    # model & output
    model = RNN(seq_len)  # other parameters just default
    out = model(x)
    print(f"out shape is {out.shape}")


if __name__ == "__main__":
    main()

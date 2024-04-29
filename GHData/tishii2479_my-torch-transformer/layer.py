from torch import Tensor, nn
import torch
import torch.nn.functional as F
from function import attention

# PyTorchにおけるカスタムレイヤーの実装
# https://www.bigdata-navi.com/aidrops/2890/


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        h: int,
        embedding_size: int,
        is_masked: bool = False,
        d_k: int = None,
        d_v: int = None
    ):
        super().__init__()
        self.h = h
        self.is_masked = is_masked
        self.d_k = embedding_size // h if d_k is None else d_k
        self.d_v = embedding_size // h if d_v is None else d_v

        self.lq = nn.Linear(embedding_size, self.d_k * h, bias=False)
        self.lk = nn.Linear(embedding_size, self.d_k * h, bias=False)
        self.lv = nn.Linear(embedding_size, self.d_v * h, bias=False)

        self.lo = nn.Linear(h * self.d_v, embedding_size, bias=False)

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor
    ):
        '''
        Q = (batch_size, sequence_length, embedding_size)
        K = (batch_size, sequence_length, embedding_size)
        V = (batch_size, sequence_length, embedding_size)

        return (batch_size, sequence_length, embedding_size)
        '''
        Q = self.lq.forward(Q)
        K = self.lk.forward(K)
        V = self.lv.forward(V)

        heads = []

        for i in range(self.h):
            head_i = attention(
                Q[:, :, i * self.d_k: (i+1) * self.d_k],
                K[:, :, i * self.d_k: (i+1) * self.d_k],
                V[:, :, i * self.d_v: (i+1) * self.d_v],
                self.is_masked
            )
            heads.append(head_i)

        return self.lo.forward(torch.cat(heads, dim=2))


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        activation: lambda x: x = F.relu
    ):
        super().__init__()
        self.l1 = nn.Linear(embedding_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, embedding_size)
        self.activation = activation

    def forward(self, x: Tensor):
        '''
        x = (batch_size, sequence_length, embedding_size)
        '''
        return self.l2.forward(self.activation(self.l1.forward(x)))

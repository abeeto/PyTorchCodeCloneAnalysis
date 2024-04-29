import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()

        # Pointwise Convolution
        self.query_conv = nn.Conv2d(in_channels=input_dim,
                                    out_channels=input_dim//8,
                                    kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=input_dim,
                                  out_channels=input_dim//8,
                                  kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=input_dim,
                                    out_channels=input_dim,
                                    kernel_size=1)

        # softmax module
        self.softmax = nn.Softmax(dim=-2)

        # 大域的な情報を足し込む際の係数
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Pointwise Convolutionを適用、チャネル数を変える
        proj_query = self.query_conv(x).view(x.shape[0], -1, x.shape[2] * x.shape[3])
        proj_query = proj_query.permute(0, 2, 1)    # (Batch, C, N)へと転置
        proj_key = self.key_conv(x).view(x.shape[0], -1, x.shape[2] * x.shape[3])

        # make attention map
        at_map = self.softmax(torch.bmm(proj_query, proj_key))
        at_map = at_map.permute(0, 2, 1)

        # make self-attention map
        proj_value = self.value_conv(x).view(x.shape[0], -1, x.shape[2] * x.shape[3])
        sa_map = torch.bmm(proj_value, at_map.permute(0, 2, 1))
        sa_map = sa_map.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        out = x + self.gamma * sa_map
        return out, at_map  # attention mapの可視化のため

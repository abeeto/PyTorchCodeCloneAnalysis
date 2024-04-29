import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=False, qkv_scale=None, dropout=0., attention_dropout=0.):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = int(embed_dim / num_heads)
        self.all_head_dim = self.head_dim * num_heads
        self.qkv = nn.Linear(
            embed_dim,
            self.all_head_dim * 3,
            bias=False if qkv_bias is False else True
        )
        self.scale = self.head_dim ** -0.5 if qkv_scale is None else qkv_scale
        self.softmax = nn.Softmax(dim=-1)  # 对最后一维做softmax
        self.proj = nn.Linear(self.all_head_dim, embed_dim)

    def transpose_multi_head(self, x):
        # x: [B, num_patches, all_head_dim]
        new_shape = list(x.shape[:-1]) + [self.num_heads, self.head_dim]
        x = x.reshape(new_shape)
        # x: [B, num_patches, num_heads, head_dim]
        x = x.permute(0, 2, 1, 3)
        # x: [B, num_heads, num_patches, head_dim]
        return x

    def forward(self, x):
        B, num_patches, _ = x.shape
        # calculate q k v
        qkv = self.qkv(x).chunk(3, -1)  # 将最后一维分3块，得到：qkv，返回的是一个列表
        q, k, v = map(self.transpose_multi_head, qkv)
        # q k v: [B, num_heads, num_patches, head_dim]
        a = torch.matmul(q, k.permute(0, 1, 3, 2))  # 转置k的最后两维
        # a: [B, num_heads, num_patches, num_patches]
        a = self.scale * a
        a = self.softmax(a)

        out = torch.matmul(a, v)
        # out: [B, num_heads, num_patches, head_dim]
        out = out.permute(0, 2, 1, 3)
        # out: [B, num_patches, num_heads, head_dim]
        out = out.reshape(B, num_patches, -1)
        # out: [B, num_patches, num_heads*head_dim]
        out = self.proj(out)
        return out


def main():
    x = torch.randn(2, 16, 96)
    print(f"input shape is {x.shape}")
    model = Attention(embed_dim=96, num_heads=4, qkv_bias=False, qkv_scale=None)
    out = model(x)
    print(f"output shape is {out.shape}")


if __name__ == "__main__":
    main()

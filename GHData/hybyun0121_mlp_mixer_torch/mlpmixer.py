import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class MlpBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim):
        super(MlpBlock, self).__init__()
        self.in_dim = in_dim
        self.mlp_dim = mlp_dim
        self.lin1 = nn.Linear(self.in_dim, self.mlp_dim)
        self.lin2 = nn.Linear(self.mlp_dim, self.in_dim)

    def forward(self, x):
        out = F.gelu(self.lin1(x))
        return self.lin2(out)

class MixerBlock(nn.Module):
    def __init__(self, H, W, patch_h, patch_w, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super(MixerBlock, self).__init__()
        self.H, self.W = H, W
        self.patch_h, self.patch_w = patch_h, patch_w
        self.hidden_dim = hidden_dim
        self.num_patches = (self.H // self.patch_h) * (self.W // self.patch_w)
        self.tokens = tokens_mlp_dim
        self.channels = channels_mlp_dim

        self.norm = nn.LayerNorm([self.num_patches, self.hidden_dim])
        self.mlpblock_1 = MlpBlock(in_dim=self.num_patches, mlp_dim=self.tokens)
        self.mlpblock_2 = MlpBlock(in_dim=self.hidden_dim, mlp_dim=self.channels)

    def forward(self, x):
        out = self.norm(x)
        out = torch.swapaxes(out, 1, 2)
        out = self.mlpblock_1(out)
        out = torch.swapaxes(out, 1, 2)
        x = out + x
        out = self.norm(x)
        return x + self.mlpblock_2(out)

class MlpMixer(nn.Module):
    def __init__(self, layout_size, num_classes, num_blocks, patch_size, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super(MlpMixer, self).__init__()
        self.H, self.W = pair(layout_size)
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.patch_h, self.patch_w = pair(patch_size)
        self.hidden_dim = hidden_dim
        self.tokens = tokens_mlp_dim
        self.channels = channels_mlp_dim

        self.conv = nn.Conv2d(self.num_classes, self.hidden_dim, kernel_size=(self.patch_h, self.patch_w),
                              stride=(self.patch_h, self.patch_w))

        self.mixerblock = MixerBlock(self.H, self.W, self.patch_h, self.patch_w, self.hidden_dim, self.tokens, self.channels)

    def forward(self, x):
        x = self.conv(x)
        x = einops.rearrange(x, 'n c h w -> n (h w) c')
        for _ in range(self.num_blocks):
            print(x.shape)
            x = self.mixerblock(x)
        return x

x = torch.randn(1, 3, 225, 225)
layout_size = (15, 15)
num_classes = 3
num_blocks = 2
patch_size = (3, 3)
hidden_dim = 3
tokens_mlp_dim = 3
channels_mlp_dim = 3

model = MlpMixer(layout_size=layout_size,
                 num_classes=num_classes,
                 num_blocks=num_blocks,
                 patch_size=patch_size,
                 hidden_dim=hidden_dim,
                 tokens_mlp_dim=tokens_mlp_dim,
                 channels_mlp_dim=channels_mlp_dim)
out = model(x)
print(out.shape)
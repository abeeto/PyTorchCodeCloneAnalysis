import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_dim, activation):
        """
        self attention mechanism
        """
        super(SelfAttention, self).__init__()
        self.channel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs: 
                x: input feature maps (B x C x W x H)
            returns:
                out: self attention value + input feature
                attention: B x N x N (N is width*height)
        """
        m_batch, channels, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batch, -1, width * height).permute(0, 2, 1) # B N C
        proj_key = self.key_conv(x).view(m_batch, -1, width*height) # B C N
        energy = torch.bmm(proj_query, proj_key) # transpose check, B N N
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batch, -1, width*height) # B C N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batch, channels, width, height)

        out = self.gamma * out + x

        return out, attention

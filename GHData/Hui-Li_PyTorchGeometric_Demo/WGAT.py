import torch
import torch_geometric
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot

import torch.nn.functional as F

# Reference: GATConv
# https://github.com/rusty1s/pytorch_geometric/blob/40a35ac3cd60413f8a5f436adf9e0e0d8b6af42c/torch_geometric/nn/conv/gat_conv.py
class WGATConv(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=1, negative_slope=0.2, dropout=0, weighted=True):
        super(WGATConv, self).__init__(aggr='add', node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.weighted = weighted

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))

        if weighted:
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels + 1))
        else:
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)

    def forward(self, x, edge_index, edge_attr=None):

        # node_nums x (heads * out_channels)
        x = torch.matmul(x, self.weight)

        out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

        # node_nums x heads x out_channels
        out = out.mean(dim=1)

        # node_nums x out_channels
        return out


    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr=None):

        # Compute attention coefficients.
        # edge_num x heads x out_channels
        x_j = x_j.view(-1, self.heads, self.out_channels)
        x_i = x_i.view(-1, self.heads, self.out_channels)

        if self.weighted and edge_attr is not None:
            # edge_attr: edge_num

            # edge_num x heads x (2 * out_channels + 1)
            alpha = (torch.cat([x_i, x_j, edge_attr.view(-1, 1).repeat(1, self.heads).view(-1, self.heads, 1)],
                               dim=-1) * self.att).sum(dim=-1)
        else:
            # [edge_num x heads x (2 * out_channels)] * [1 x heads x (2 * out_channels)] -> edge_num x heads
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        # edge_num x heads
        alpha = torch_geometric.utils.softmax(src=alpha, index=edge_index_i, num_nodes=size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # [edge_num x heads x out_channels] * [edge_num x heads x 1] -> [edge_num x heads x out_channels]
        result = x_j * alpha.view(-1, self.heads, 1)

        return result
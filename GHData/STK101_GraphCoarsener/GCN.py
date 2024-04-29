import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# class GCN(torch.nn.Module):
#     def __init__(self, num_features, hidden_channels,num_classes):
#         super(GCN, self).__init__()
#         torch.manual_seed(42)

#         self.conv1 = GCNConv(num_features, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         self.out = Linear(hidden_channels, num_classes)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         # x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         # x = F.dropout(x, p=0.5, training=self.training)
#         x = F.softmax(self.out(x), dim=1)
#         return x

# class GCN(torch.nn.Module):
#     def __init__(self, num_features,hidden_channels,num_classes):
#         super().__init__()
#         self.conv1 = GCNConv(num_features, 16)
#         self.conv2 = GCNConv(16, num_classes)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)

class GCN(torch.nn.Module):
    def __init__(self, num_features,hidden_channels,num_classes):
        super(GCN, self).__init__()

        self.gc1 = GCNConv(num_features, 16)
        self.gc2 = GCNConv(16, num_classes)
        self.dropout = 0.5

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
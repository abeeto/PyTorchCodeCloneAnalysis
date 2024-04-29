import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, A_tilde,
                 activation=None,
                 featureless=False,
                 dropout_rate=0.):
        super(GraphConvolution, self).__init__()
        self.A_tilde = A_tilde
        self.featureless = featureless
        self.Theta = nn.Parameter(torch.randn(input_dim, output_dim))
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        #x = self.dropout(x)
        if self.featureless:
            pre_sup = self.Theta
        else:
            pre_sup = x @ self.Theta

        out = self.A_tilde @ pre_sup

        if self.activation is not None:
            out = self.activation(out)

        self.embedding = out
        return out


class GCN(nn.Module):
    def __init__(self, input_dim, A_tilde,
                 dropout_rate=0.,
                 num_classes=10):
        super(GCN, self).__init__()

        # GraphConvolution
        self.layer1 = GraphConvolution(input_dim, 200, A_tilde, activation=nn.ReLU(), featureless=True,
                                       dropout_rate=dropout_rate)
        self.layer2 = GraphConvolution(200, num_classes, A_tilde, dropout_rate=dropout_rate)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out

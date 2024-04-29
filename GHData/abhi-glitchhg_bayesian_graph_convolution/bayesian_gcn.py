from torch_geometric.nn import GCNConv

from .linear import GaussianLinear

class BGCNConv(GCNConv):
    """
    GCN with bayesian linear layer.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv, self).__init__(in_channels = in_channels, out_channels=out_channels,improved=improved, cached = cached, add_self_loops= add_self_loops, normalize=normalize,
                                      bias= bias, kwargs=kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved= improved 
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize 
        self.bias = bias

        self._cached_edge_index = None
        self._cached_adj_t = None
        self.lin = GaussianLinear(in_channels, out_channels, bias=True,)

    def get_pw(self):
        return self.lin.get_pw()

    def get_qw(self):
        return self.lin.get_qw()


import torch
class LinearTransformation(object):
    r"""Transforms node positions with a square transformation matrix computed
    offline.

    Args:
        matrix (Tensor): tensor with shape :math:`[D, D]` where :math:`D`
            corresponds to the dimensionality of node positions.
    """

    def __init__(self, matrix):
        assert matrix.dim() == 2, (
            'Transformation matrix should be two-dimensional.')
        assert matrix.size(0) == matrix.size(1), (
            'Transformation matrix should be square. Got [{} x {}] rectangular'
            'matrix.'.format(*matrix.size()))

        self.matrix = matrix

    def __call__(self, data):
        pos = data.pos.view(-1, 1) if data.pos.dim() == 1 else data.pos
        wss_coord = data.wss_coord.view(-1, 1) if data.wss_coord.dim() == 1 else data.wss_coord
        #norm=data.norm.view(-1, 1) if data.norm.dim() == 1 else data.norm
        assert pos.size(-1) == self.matrix.size(-2), (
            'Node position matrix and transformation matrix have incompatible '
            'shape.')
        assert wss_coord.size(-1) == self.matrix.size(-2), (
            'Node WSS matrix and transformation matrix have incompatible '
            'shape.')
        
        data.pos = torch.matmul(pos, self.matrix.to(pos.dtype).to(pos.device))
        #data.norm = torch.matmul(norm, self.matrix.to(norm.dtype).to(norm.device))
        data.wss_coord = torch.matmul(wss_coord, self.matrix.to(wss_coord.dtype).to(wss_coord.device))
        print('Rotation Done')
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.matrix.tolist())
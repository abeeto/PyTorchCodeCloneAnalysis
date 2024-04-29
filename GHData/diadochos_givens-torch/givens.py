import itertools
import torch
import torch.nn as nn


def G_transpose(D, i, j, theta):
    """Generate Givens rotation matrix.
    >>> G_transpose(2, 0, 1, torch.FloatTensor([[3.1415 / 2]]))
    tensor([[ 4.6329e-05,  1.0000e+00],
            [-1.0000e+00,  4.6329e-05]])
    """
    R = torch.eye(D)
    s, c = torch.sin(theta), torch.cos(theta)
    R[i, i] = c
    R[j, j] = c
    R[i, j] = s
    R[j, i] = -s
    return R


class Rotation(nn.Module):
    def __init__(self, D):
        """
        >>> # Initialized as an identity.
        >>> A, R = torch.eye(3), Rotation(3)
        >>> torch.all(A.eq(R(A))).item()
        True
        """
        super().__init__()
        self.D = D
        self.theta = torch.zeros((len(list(itertools.combinations(range(self.D), 2))), ))

    def forward(self, x):
        """Apply rotation.
        >>> A, R = torch.eye(3), Rotation(3)
        >>> R.theta = torch.FloatTensor([3.1415 / 2, 0., 0.])
        >>> R(A)
        tensor([[ 4.6329e-05,  1.0000e+00,  0.0000e+00],
                [-1.0000e+00,  4.6329e-05,  0.0000e+00],
                [ 0.0000e+00,  0.0000e+00,  1.0000e+00]])
        """
        for idx, (i, j) in enumerate(itertools.combinations(range(self.D), 2)):
            x = torch.matmul(x, G_transpose(self.D, i, j, self.theta[idx]))
        return x

    def reverse(self, x):
        """Apply reverse rotation.
        >>> A, R = torch.eye(3), Rotation(3)
        >>> R.weight = torch.FloatTensor([1., 2., 3.])
        >>> torch.any(A.eq(R(A))).item()
        True
        >>> torch.all(A.eq(R.reverse(R(A)))).item()
        True
        """
        for idx, (i, j) in reversed(list(enumerate(itertools.combinations(range(self.D), 2)))):
            x = torch.matmul(x, G_transpose(self.D, i, j, -self.theta[idx]))
        return x


if __name__ == '__main__':
    import doctest
    doctest.testmod()

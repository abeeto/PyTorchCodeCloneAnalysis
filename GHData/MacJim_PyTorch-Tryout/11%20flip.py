# Flip a tensor.
# This could be useful in image processing.

import torch


# tensor([[0., 1., 2., 3.],
#         [4., 5., 6., 7.]])
a = torch.Tensor(list(range(8))).view(2, -1)
print(a)

# tensor([[4., 5., 6., 7.],
#         [0., 1., 2., 3.]])
b = torch.flip(a, [0])
print(b)

# tensor([[3., 2., 1., 0.],
#         [7., 6., 5., 4.]])
c = torch.flip(a, [1])
print(c)

# tensor([[7., 6., 5., 4.],
#         [3., 2., 1., 0.]])
d = torch.flip(a, [0, 1])    # Flip on multiple axes.
print(d)

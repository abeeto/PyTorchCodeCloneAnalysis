import torch

x = torch.tensor([[1., 2., 3., 4., ],
                  [5., 6., 7., 8., ],
                  [9., 10., 11., 12.]], requires_grad=True)

function = 20 * (x ** 3).sum()

function.backward()  # do grad for f = SUM_i_j ( 20*x_i_j^3 ) = > f' = 20 * 3 * SUM_i_j (2 * x_i_j^2 )

print(x)
print(x.grad)
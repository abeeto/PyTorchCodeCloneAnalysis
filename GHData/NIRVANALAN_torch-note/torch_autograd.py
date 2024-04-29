import torch
from torch.autograd import Variable

T, D = 3, 4
y0 = Variable(torch.randn(D))
x = Variable(torch.randn(T, D))
w = Variable(torch.randn(D))

y = [y0]
for t in range(T):
	prev_y = y[-1]
	next_y = (prev_y + x[t]) * w
	y.append(next_y)
print(y)

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

x,y = torch.autograd.Variable(x), Variable(y)

plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

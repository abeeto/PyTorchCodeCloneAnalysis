import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0)

# Dummy training data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.1 * torch.randn(x.size())
x, y = Variable(x), Variable(y)
# print(x)
# print(y)

# a 1 hidden layer neural network
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.elu(self.hidden(x))
        output = self.out(x)
        return output

net = Net(n_feature=1, n_hidden=10, n_output=1)

optimizer = torch.optim.Adam(net.parameters(), lr=0.4)
loss_fn = torch.nn.SmoothL1Loss()

num_iter = 200
losses = []
for t in range(num_iter):
    prediction = net(x)
    loss = loss_fn(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (t + 1) % 10 == 0:
        losses.append(loss.data[0])
        print("Timestep: " + str(t + 1) + ", loss = " + str(loss.data[0]) + "....")

# plt.plot(losses)
# plt.show()
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=2)
plt.savefig("./Figures/regression_out.png")

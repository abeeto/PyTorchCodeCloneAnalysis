import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# fake data
data = torch.ones(100, 2)

x0 = torch.normal(2 * data, 1)
y0 = torch.zeros(100)

x1 = torch.normal(5 * data, 1)
y1 = torch.ones(100)

x = torch.cat((x0, x1), dim=0).type(torch.FloatTensor)
y = torch.cat((y0, y1), dim=0).type(torch.LongTensor)

x, y = Variable(x), Variable(y)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.elu(self.hidden(x))
        # output = F.sigmoid(self.out(x))
        output = self.out(x)
        return output

net = Net(n_feature=2, n_hidden=10, n_output=2)

optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
loss_fn = torch.nn.CrossEntropyLoss()

num_iter = 80
for t in range(num_iter):
    out = net(x)
    loss = loss_fn(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (t + 1) % 4 == 0:
        prediction = torch.max(out, dim=1)[1].data.numpy()
        actual_y = y.data.numpy()
        accuracy = sum(prediction == actual_y) / 200
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=prediction)
        plt.savefig("Figures/Classifier/test_" + str(t) + ".png")
        print("Timestep: " + str(t + 1) + ", accuracy = " + str(accuracy * 100) + "...")

plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=prediction)
plt.show()

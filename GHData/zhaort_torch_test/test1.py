import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1)).type(torch.LongTensor)

print(x.shape)
# plt.scatter(x.numpy(), y.numpy())
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        # x = torch.nn.ReLU(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(2, 20, 2)

net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2),
)
print(net)
print(net2)

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_fn = torch.nn.CrossEntropyLoss()

for t in range(100):
    prediction = net(x)
    loss = loss_fn(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 10 == 0:
        print(t)
        print(loss.item())

torch.save(net2, 'net.pkl')  # entire net
torch.save(net2.state_dict(), 'net_params.pkl')  # parameters

net_test = torch.load('net.pkl')
print(net_test)

net_test2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2),
)

net_test2.load_state_dict(torch.load('net_params.pkl'))

print(torch.load('net_params.pkl'))

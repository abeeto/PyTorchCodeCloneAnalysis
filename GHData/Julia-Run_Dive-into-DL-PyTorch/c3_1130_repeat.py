import torch
import numpy as np

# data
true_w = [4.3, -5]
true_b = 9
num_examples = 1000
num_inputs = len(true_w)
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = features[:, 0] * true_w[0] + features[:, 1] * true_w[1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, labels.size()), dtype=torch.float)

# data
import torch.utils.data as Data

batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

for x, y in data_iter:
    print(x)
    print(y)
    break
# model
import torch.nn as nn

net = nn.Sequential(nn.Linear(num_inputs, 1))

# init
nn.init.normal_(net[0].weight, mean=0, std=0.01)
nn.init.constant_(net[0].bias, val=0)

# loss
loss = nn.MSELoss()

# SGD
import torch.optim as op

optimizer = op.SGD(net.parameters(), lr=0.03)

# trainning
echos = 3
for i in range(1, echos):
    for x, y in data_iter:
        yhat = net(x)
        l = loss(yhat, y.view(-1, 1))  # 保证y的size。否则会出错
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('echo =  %d, loss = %s' % (i, l.item()))
p = [param for param in net.parameters()]
print(p)
print(true_w, true_b, sep='     ')

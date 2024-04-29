import torch
from torch.utils import data
from torch import nn


def generate_data(w, b, num_examples):
    """Return a random matrix X of examples and corresponding
    labels y generated using weights w and bias b. Labels are
    given additional noise."""

    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    noise = torch.normal(0, 0.01, y.shape)
    y += noise
    return X, y.reshape((-1, 1))


def load_array(data_arrays, batch_size, is_train=True):
    """Return a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


real_weights = torch.tensor([5.2, -1.7])
real_bias = 2.77

features, labels = generate_data(real_weights, real_bias, 500)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

net = nn.Sequential(nn.Linear(2, 1))

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.01)

num_epochs = 10
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'Epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('\nError in estimating w:', real_weights - w.reshape(real_weights.shape))
b = net[0].bias.data
print('Error in estimating b:', real_bias - b)

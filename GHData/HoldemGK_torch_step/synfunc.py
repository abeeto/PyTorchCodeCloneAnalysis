import torch
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['figure.figsize'] = (13.0, 5.0)

x_train = torch.rand(100)
x_train = x_train * 20.0 - 10.0
y_train = torch.sin(x_train)
#plt.plot(x_train.numpy(), y_train.numpy(), 'o')
plt.title('$y = sin(x)$')

noise = torch.randn(y_train.shape) / 5.
#plt.plot(x_train.numpy(), noise.numpy(), 'o')
plt.axis([-10, 10, -1, 1])
plt.title('Gaussian noise')

y_train = y_train + noise
#plt.plot(x_train.numpy(), y_train.numpy(), 'o')
plt.title('noisy sin(x)')
plt.xlabel('x_train')
plt.ylabel('y_train')

x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

tmp = torch.Tensor([1, 2, 3])
print(tmp)
print(tmp.unsqueeze(1))

# Validation dataset
x_validation = torch.linspace(-10, 10, 100)
y_validation = torch.sin(x_validation.data)
#plt.plot(x_validation.numpy(), y_validation.numpy(), 'o')
plt.title('sin(x)')
plt.xlabel('x_validation')
plt.ylabel('y_validation')

x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)

# Model construction


class SineNet(torch.nn.Module):
    def __init__(self, n_hiden_neurons):
        super(SineNet, self).__init__()
        self.fc1 = torch.nn.Linear(1, n_hiden_neurons)
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hiden_neurons, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


sine_net = SineNet(50)

# Prediction


def predict(net, x, y):
    y_pred = net.forward(x)
    plt.plot(x.numpy(), y.numpy(), 'o', label='Ground truth')
    plt.plot(x.numpy(), y_pred.data.numpy(), 'o', c='r', label='Prediction')
    plt.legend(loc='upper left')
    plt.xlabel('$x$')
    plt.ylabel('$x$')


# Optimizer
optimizer = torch.optim.Adam(sine_net.parameters(), lr=0.01)

# Loss function


def loss(pred, target):
    squares = (pred - target) ** 2
    return squares.mean()


# Trainig procedure
for epoch_index in range(2000):
    optimizer.zero_grad()

    y_pred = sine_net.forward(x_train)
    loss_val = loss(y_pred, y_train)
    loss_val.backward()
    optimizer.step()
predict(sine_net, x_validation, y_validation)
plt.show()

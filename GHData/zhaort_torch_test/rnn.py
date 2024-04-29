import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as Data

EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = False
HIDDEN_SIZE = 64

train_data = torchvision.datasets.MNIST(root='./mnist', train=True,
                                        transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist', train=False, transform=torchvision.transforms.ToTensor())
test_x = test_data.data.type(torch.FloatTensor)[:200]/255.
test_y = test_data.targets.numpy().squeeze()[:200]


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(HIDDEN_SIZE, 10)

    def forward(self, x):
        out, hidden_prev = self.rnn(x, None)
        out = self.out(out[:, -1, :])  # shape
        return out


rnn = RNN()
print(rnn)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()
try:
    rnn.load_state_dict(torch.load('rnn'))
except Exception as e:
    print(e)


for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        x = x.view(-1, 28, 28)
        output = rnn(x)
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn(test_x).detach()
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()  # shape
            accuracy = sum(pred_y == test_y) / test_y.size
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '|test accuracy: %.2f' %accuracy)
            torch.save(rnn.state_dict(), 'rnn')
torch.save(rnn.state_dict(), 'rnn')
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()  # shape
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')

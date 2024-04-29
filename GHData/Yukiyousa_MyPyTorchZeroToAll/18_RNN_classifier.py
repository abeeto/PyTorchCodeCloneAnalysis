import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(1)

EPOCH = 5
BATCH_SIZE = 64
TIME_STEP = 28  # rnn time step / image height
INPUT_SIZE = 28  # rnn input size / image width
LR = 0.001
DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=torchvision.transforms.ToTensor())
# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255
test_y = test_data.test_labels[:2000]


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=28,
            hidden_size=64,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers,  batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, data in enumerate(train_loader):
        x = data[0].view(-1, 28, 28)
        optimizer.zero_grad()
        output = rnn(x)
        batch_loss = loss(output, data[1])
        batch_loss.backward()
        optimizer.step()

        if step % 100 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            # accuracy = (pred_y == test_y).sum() / float(test_y.size(0))
            accuracy = sum((pred_y == test_y)) / test_y.size(0)
            print('Epoch:%d/%d | train loss: %.4f | test accuracy: %.2f' % (epoch, EPOCH, batch_loss.item(), accuracy))








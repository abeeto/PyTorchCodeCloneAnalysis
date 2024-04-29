import torch
import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self, input_size, classes):
        super(NN, self).__init__()  # super calls the init method of the parent class(nn.Module)
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, classes)

    def forward(self, inp):
        out = F.relu(self.fc1(inp))
        out = self.fc2(out)

        return out


class CNN(nn.Module):
    def __init__(self, channels=1, classes=10):
        super(CNN, self).__init__()  # super calls the init method of the parent class(nn.Module)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16 * 7 * 7, classes)  # out channels of conv2 = 16

    def forward(self, inp):
        out = F.relu(self.conv1(inp))
        out = self.pool(out)
        out = F.relu(self.conv2(out))
        out = self.pool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)

        return out


class RNN(nn.Module):
    def __init__(self, size, hidden, layers, seq_len, classes, device):  # size is the number of time steps
        super(RNN, self).__init__()
        self.hidden = hidden
        self.layers = layers
        self.device = device
        self.rec = nn.RNN(input_size=size, hidden_size=hidden, num_layers=layers, batch_first=True)
        # batch_first = True in case, first axis represents size of batch
        self.fc = nn.Linear(hidden * seq_len, classes)

    def forward(self, inp):
        h0 = torch.zeros(self.layers, inp.shape[0], self.hidden).to(self.device)  # initial hidden state, 2x64x256

        out, state = self.rec(inp, h0)  # hidden state is ignored
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)

        return out


# 64 samples in a batch
model = NN(784, 10)
data = torch.rand(64, 784)

# model = CNN()
# data = torch.rand(64, 1, 28, 28)  # 1 channel

# print(model(data).shape)

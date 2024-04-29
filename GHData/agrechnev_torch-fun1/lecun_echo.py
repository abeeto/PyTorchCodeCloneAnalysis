# By IT-JIM, 16-Apr-2021

import sys

import numpy as np
import matplotlib.pyplot as plt

import torch

import lecun_plot
from res.sequential_tasks import EchoData


########################################################################################################################
class SimpleRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, nonlinearity='relu', batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x, hidden = self.rnn(x, hidden)
        x = self.linear(x)
        return x, hidden


########################################################################################################################
class Trainer:
    def __init__(self):
        # Misc
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.n_epochs = 10

        # Data
        batch_size = 5
        echo_step = 3
        series_length = 20_000
        bptt_t = 20
        self.train_data = EchoData(
            echo_step=echo_step,
            batch_size=batch_size,
            series_length=series_length,
            truncated_length=bptt_t,
        )
        self.train_size = len(self.train_data)

        self.test_data = EchoData(
            echo_step=echo_step,
            batch_size=batch_size,
            series_length=series_length,
            truncated_length=bptt_t,
        )
        self.test_size = len(self.test_data)

        # Model etc
        self.model = SimpleRNN(1, 4, 1).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.001)
        self.hidden = None

    def train(self):
        self.model.train()
        correct = 0
        for idx_b in range(self.test_size):
            x, y = self.test_data[idx_b]
            x = torch.from_numpy(x).float().to(self.device)
            y = torch.from_numpy(y).float().to(self.device)
            self.optimizer.zero_grad()
            if self.hidden is not None:
                self.hidden.detach_()
            out, self.hidden = self.model(x, self.hidden)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()
            pred = (torch.sigmoid(out) > 0.5)
            correct += (pred == y.byte()).int().sum().item()

        return correct / (self.test_size * 100), loss.item()

    def test(self):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for idx_b in range(self.test_size):
                x, y = self.test_data[idx_b]
                x = torch.from_numpy(x).float().to(self.device)
                y = torch.from_numpy(y).float().to(self.device)
                out, self.hidden = self.model(x, self.hidden)
                loss = self.criterion(out, y)
                pred = (torch.sigmoid(out) > 0.5)
                correct += (pred == y.byte()).int().sum().item()

        return correct / (self.test_size * 100), loss.item()

    def training_loop(self):
        print('TEST :', self.test())
        for epoch in range(self.n_epochs):
            acc_train, loss_train = self.train()
            print(acc_train, loss_train)
        print('TEST :', self.test())


########################################################################################################################
def main():
    torch.manual_seed(1)
    np.random.seed(1)
    lecun_plot.set_default()
    trainer = Trainer()
    trainer.training_loop()
    # print(trainer.test())
    # print(trainer.train())
    # print(trainer.train())
    # print(trainer.train())


########################################################################################################################
if __name__ == '__main__':
    main()

# By IT-JIM, 12-Apr-2021
import sys
import collections


import numpy as np
import matplotlib.pyplot as plt

import torch

import lecun_plot

from res.sequential_tasks import TemporalOrderExp6aSequence as QRSU


########################################################################################################################
class SimpleRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, nonlinearity='relu', batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = self.rnn(x)[0]
        x = self.linear(h)
        return x


########################################################################################################################
class SimpleLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = self.lstm(x)[0]
        x = self.linear(h)
        return x


########################################################################################################################
class Trainer:
    def __init__(self):
        self.difficulty = QRSU.DifficultyLevel.EASY
        self.batch_size = 32
        self.gen_train = QRSU.get_predefined_generator(self.difficulty, self.batch_size)
        self.gen_test = QRSU.get_predefined_generator(self.difficulty, self.batch_size)
        self.max_epochs = 100
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        size_input = self.gen_train.n_symbols
        size_hidden = 4
        size_output = self.gen_train.n_classes
        # self.model = SimpleRNN(size_input, size_hidden, size_output).to(self.device)
        self.model = SimpleLSTM(size_input, size_hidden, size_output).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.001)

        self.hist_train = None
        self.hist_test = None

        print('===== PARAMETER GROUPS ====')
        for pg in list(self.model.parameters()):
            print(pg.size())
        print('===========================')

    def training_loop(self, verbose=True):
        """The complete training loop over epochs"""
        self.hist_train = {'loss': [], 'acc': []}
        self.hist_test = {'loss': [], 'acc': []}
        for epoch in range(self.max_epochs):
            # Train
            nc_train, loss_train = self.train()
            acc_train = nc_train / (len(self.gen_train) * self.batch_size) * 100
            self.hist_train['loss'].append(loss_train)
            self.hist_train['acc'].append(acc_train)
            # Test
            nc_test, loss_test = self.test()
            acc_test = nc_test /  (len(self.gen_test) * self.batch_size) * 100
            self.hist_test['loss'].append(loss_test)
            self.hist_test['acc'].append(acc_test)

            if verbose or epoch == self.max_epochs - 1:
                print('[Epoch {} / {}]   loss :  {:2.2f}, acc : {:2.2f}%  - test loss : {:2.2f}, acc : {:2.2f}%'
                      .format(epoch + 1, self.max_epochs, loss_train, acc_train, loss_test, acc_test))

        # Plot stuff
        if False:
            fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
            for ax, metric in zip(axes, ['loss', 'acc']):
                ax.plot(self.hist_train[metric])
                ax.plot(self.hist_test[metric])
                ax.set_xlabel('epoch', fontsize=12)
                ax.set_ylabel(metric, fontsize=12)
                ax.legend(['Train', 'Test'], loc='best')
            plt.show()

    def train(self):
        """One training epoch !"""
        self.model.train()
        num_correct = 0
        for b_idx in range(len(self.gen_test)):
            x, y = self.gen_test[b_idx]
            x = torch.from_numpy(x).float().to(self.device)
            y = torch.from_numpy(y).long().to(self.device)
            out = self.model(x)[:, -1, :]  # Last element in each sequence !!!
            target = y.argmax(dim=1)
            loss = self.criterion(out, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred = out.argmax(dim=1)
            num_correct += (pred == target).sum().item()
        return num_correct, loss.item()

    def test(self):
        """One test epoch"""
        self.model.eval()
        num_correct = 0
        with torch.no_grad():
            for b_idx in range(len(self.gen_test)):
                x, y = self.gen_test[b_idx]
                x = torch.from_numpy(x).float().to(self.device)
                y = torch.from_numpy(y).long().to(self.device)
                out = self.model(x)[:, -1, :]  # Last element in each sequence !!!
                target = y.argmax(dim=1)
                loss = self.criterion(out, target)
                pred = out.argmax(dim=1)
                num_correct += (pred == target).sum().item()
        return num_correct, loss.item()

    def eval(self):
        """Eval on diferent data ?"""
        # Define a dictionary that maps class indices to labels
        class_idx_to_label = {0: 'Q', 1: 'R', 2: 'S', 3: 'U'}

        gen_data = QRSU.get_predefined_generator(self.difficulty, seed=9001)
        counter = collections.Counter()
        correct, incorrect = [], []

        self.model.eval()
        with torch.no_grad():
            for b_idx in range(len(gen_data)):
                x, y = gen_data[b_idx]
                bsize = x.shape[0]
                x = torch.from_numpy(x).float().to(self.device)
                y = torch.from_numpy(y).long().to(self.device)
                out = self.model(x)
                target = y.argmax(dim=1)

                x_decoded = gen_data.decode_x_batch(x.cpu().numpy())
                y_decoded = gen_data.decode_y_batch(y.cpu().numpy())
                seq_end = torch.tensor([len(s) for s in x_decoded]) - 1

                out2 = out[torch.arange(bsize).long(), seq_end, :]
                pred = out2.argmax(dim=1)
                pred_decoded = [class_idx_to_label[yy.item()] for yy in pred]

                counter.update(y_decoded)
                for i, (t, p) in enumerate(zip(y_decoded, pred_decoded)):
                    if t == p:
                        correct.append((x_decoded[i], t, p))
                    else:
                        incorrect.append((x_decoded[i], t, p))
                print(correct)


########################################################################################################################
def main():
    torch.manual_seed(1)
    lecun_plot.set_default()
    trainer = Trainer()
    trainer.training_loop()
    # trainer.eval()


########################################################################################################################
if __name__ == '__main__':
    main()

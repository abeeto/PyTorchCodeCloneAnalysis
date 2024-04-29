import torch
import torch.nn as nn
import torch.optim as optim
import data
import hyperparameters
import helpers
import random
import LossLogger
import train_test_split

class LanguageModel:
    def __init__(self):
        self.hps = hyperparameters.hps
        # create training and test datasets
        train_test_split.create_train_test()

        # inputs are letters, output is category (english, french, or spanish)
        self.model = nn.LSTM(self.hps['onehot_length'], 3, self.hps['lstm_layers'], dropout=self.hps['dropout'])
        self.training_set = data.get_pairs('datasets/training_set.csv')
        self.test_set = data.get_pairs('datasets/test_set.csv')

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hps['learning_rate'])

        self.batch_size = self.hps['batch_size']

        self.loss_logger = LossLogger.LossLogger()
        self.loss_logger.clear('losses/losses.csv')

    def train(self):
        self.model.train()
        epoch_losses = []
        # train on as many pairs are in the training_set
        for _ in range(len(self.training_set) // self.batch_size):
            batch = random.sample(self.training_set, self.batch_size)
            # keep losses from each batch to be averaged later
            batch_losses = []
            for pair in batch:
                xs, y = helpers.pair_to_xy(pair)
                # make a tensor for the whole sequence
                xs = torch.stack(xs)
                y_pred, _ = self.model(xs)
                # we only want the final prediction of the sequence
                y_pred = y_pred[-1]
                seq_loss = self.criterion(y_pred, y)
                batch_losses.append(seq_loss)
            # get the mean of all losses from the batch of names
            batch_loss = torch.mean(torch.stack(batch_losses))

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            epoch_losses.append(batch_loss)
        epoch_loss = torch.mean(torch.stack(epoch_losses)).item()
        return epoch_loss

    def test(self):
        self.model.eval()
        with torch.no_grad():
            test_losses = []
            for pair in self.test_set:
                xs, y = helpers.pair_to_xy(pair)
                # make a tensor for the whole sequence
                xs = torch.stack(xs)
                y_pred, _ = self.model(xs)
                # we only want the final prediction of the sequence
                y_pred = y_pred[-1]
                seq_loss = self.criterion(y_pred, y)
                test_losses.append(seq_loss)
            test_loss = torch.mean(torch.stack(test_losses)).item()
            return test_loss

    def save(self):
        torch.save(self.model.state_dict(), 'models/model.pt')

    def log(self, train_loss, test_loss):
        self.loss_logger.write_losses(train_loss, test_loss, 'losses/losses.csv')

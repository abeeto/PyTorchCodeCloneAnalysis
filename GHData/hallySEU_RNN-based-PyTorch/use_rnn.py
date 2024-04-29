#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
 Filename @ use_rnn.py
 Author @ huangjunheng
 Create date @ 2018-06-17 16:23:27
 Description @ 
"""
import torch
import torch.nn as nn

import data_processor
from data_processor import DataProcessor
from config import Config


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out


class UseRNN(object):
    """
        数据处理
        """

    def __init__(self):
        self.config = Config()
        # Hyper-parameters

        input_size, num_classes = data_processor.cal_model_para(self.config.training_file)

        self.model = RNN(input_size, self.config.num_hidden,
                         self.config.num_layers, num_classes).to(device)

        self.loss_and_optimizer()

    def loss_and_optimizer(self):
        """
        Loss and optimizer
        :return: 
        """
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def train(self):
        """
        train
        :return: 
        """
        print 'Start training model.'
        train_loader = DataProcessor(self.config.training_file, self.config.batch_size).load()
        total_step = len(train_loader)
        for epoch in range(self.config.training_epoch):
            for i, (features, labels) in enumerate(train_loader):
                features = features.to(device)
                _, labels = torch.max(labels, 1) # 元组第一个维度为最大值，第二个维度为最大值的索引
                labels = labels.to(device)

                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                # Backward and optimize
                self.optimizer.zero_grad()  # 清空梯度缓存
                loss.backward()  # 反向传播，计算梯度
                self.optimizer.step()  # 利用梯度更新模型参数

                if (i + 1) % 100 == 0:
                    print 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'\
                        .format(epoch + 1, self.config.training_epoch, i + 1, total_step, loss.item())

        # Save the model checkpoint
        print 'Start saving model to "%s".' % self.config.save_model_path
        torch.save(self.model.state_dict(), self.config.save_model_path)

    def test(self, load_model=False):
        """
        test
        :param load_model: 
        :return: 
        """
        if load_model:
            print 'Start loading model from "%s"' % self.config.load_model_path
            self.model.load_state_dict(torch.load(self.config.load_model_path))

        test_loader = DataProcessor(self.config.test_file, self.config.batch_size).load()

        with torch.no_grad():
            correct = 0
            total = 0
            for features, labels in test_loader:
                features = features.to(device)
                _, labels = torch.max(labels, 1)
                labels = labels.to(device)

                outputs = self.model(features)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print 'Test Accuracy of the model: {} %'.format(100 * correct / total)

    def main(self):
        """
        main
        :return: 
        """
        self.train()
        self.test(load_model=True)

if __name__ == '__main__':
    rnn = UseRNN()
    rnn.main()




import torch
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import datetime

# Tensorboard 관련
from torch.utils.tensorboard import SummaryWriter

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1) 

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

class SubmittedApp():
    def import_dataset(self):
        data = pd.read_csv("weight-height.csv")
        x, y =  data['Height'], data['Weight']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        self.dataset = {
            'x_train' : torch.Tensor(x_train.to_numpy().reshape(len(x_train), 1)),
            'x_test' : torch.Tensor(x_test.to_numpy().reshape(len(x_test), 1)),
            'y_train' : torch.Tensor(y_train.to_numpy().reshape(len(y_train), 1)),
            'y_test' : torch.Tensor(y_test.to_numpy().reshape(len(y_test), 1)),
        }

    def __init__(self):
        self.timestamp = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        try:
            self.model = torch.load("model.pt")
        except:
            self.model = Model()
            self.opt_type = "adam"
            self.criterion = torch.nn.MSELoss(reduction='mean')
            if self.opt_type == "adam":
                self.lr = 0.01
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            else:
                self.lr = 0.000001
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

            self.epochs = 20000
            self.import_dataset()

            # Tensorboard의 주요 객체인 Writer
            # log path 에 해당하는 폴더를 만든다.
            log_dir = "./logs/{}".format(self.timestamp)
            self.writer = SummaryWriter(log_dir)

            f = open("{}/info.txt".format(log_dir), 'w')
            f.write("criterion : {}\n".format("MSE"))
            f.write("optimizer : {}\n".format(self.opt_type))
            f.write("learning rate : {}\n".format(self.lr))
            f.write("epochs : {}\n".format(self.epochs))
            f.close()
            self.train(self.dataset['x_train'], self.dataset['y_train'])
            self.writer.add_graph(self.model, self.dataset['x_train'])

    def run(self, input_tensor: torch.Tensor) -> torch.Tensor:
        print("input size : ({}, {})".format(input_tensor.shape[0], input_tensor.shape[1]))
        return self.model(input_tensor)

    def train(self, x_data, y_data):
        for epoch in range(self.epochs):
            y_pred = self.model(x_data)
            loss = self.criterion(y_pred, y_data)
            print(f'Epoch: {epoch} | Loss: {loss.item()} ')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % (self.epochs//10) == 0:
                self.writer.add_scalar('loss', loss.item(), epoch)
                #self.writer.add_scalar('learning_rate', self.lr , epoch)
                for i, param in enumerate(self.model.parameters()):
                    weight = np.array(param.data.numpy().reshape((1,)))
                    self.writer.add_histogram('weight_hist/{}'.format(i), weight, epoch)
                    self.writer.add_scalar('weight_val/{}'.format(i), weight, epoch)
                    #self.writer.add_scalar('weight_val/total/{}'.format(i), weight, epoch)

    def metric(self, inferred_tensor: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        return torch.mean((inferred_tensor - ground_truth)**2)


if __name__ == "__main__":
    app = SubmittedApp()
    app.import_dataset()
    y = app.run(app.dataset['x_test'])
    l = app.metric(y, app.dataset['y_test'])
    print("loss : ", l.item())
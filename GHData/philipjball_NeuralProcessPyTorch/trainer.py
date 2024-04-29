import torch
from neuralprocess import NeuralProcess, NeuralProcessLoss

class NeuralProcessTrainer(object):

    def __init__(self, neural_process: NeuralProcess, datagenerator, num_epochs: int, optimizer=torch.optim.Adam, lr=3e-4):
        self.neural_process = neural_process
        self.datagenerator = datagenerator
        self.num_epochs = num_epochs
        self.loss = NeuralProcessLoss()
        self.optimizer = optimizer(self.neural_process.parameters(), lr=lr)

    def train(self):
        x_context, x_target, y_context, y_target = self.datagenerator.create_training_set()
        self.neural_process.train()
        for i in range(self.num_epochs):
            epoch_loss = 0
            epoch_mse = 0
            for x_c, x_t, y_c, y_t in zip(x_context, x_target, y_context, y_target):
                self.optimizer.zero_grad()
                loss = self.loss(self.neural_process, x_c, y_c, x_t, y_t)
                epoch_loss += loss
                epoch_mse += loss.mse_loss
                loss.backward()
                self.optimizer.step()
            print('Epoch:{} Loss: {:.4f} Acc: {:.4f}'.format(
                    i, epoch_loss, epoch_mse))
        
        self.neural_process.eval()

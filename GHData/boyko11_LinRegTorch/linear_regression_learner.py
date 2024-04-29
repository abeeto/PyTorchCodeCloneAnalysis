import numpy as np
import torch


class LinearRegressionLearner():

    def __init__(self, theta_size, learning_rate=0.00001):
        self.theta_tensor = torch.randn(theta_size, 1, requires_grad=True)
        self.loss_function = torch.nn.L1Loss()
        self.learning_rate = learning_rate
        self.loss_history = []

    def predict(self, features_tensor):

        return torch.mm(features_tensor, self.theta_tensor)

    def calculate_loss(self, predictions_tensor, labels_tensor):

        return self.loss_function(predictions_tensor, labels_tensor)

    def train(self, features_tensor, labels_tensor, epochs=1000):

        for i in range(epochs):
            predictions = self.predict(features_tensor)
            loss = self.calculate_loss(predictions, labels_tensor)
            self.loss_history.append(loss)
            loss.backward()
            with torch.no_grad():
                self.theta_tensor -= self.theta_tensor.grad * self.learning_rate
                self.theta_tensor.grad.zero_()





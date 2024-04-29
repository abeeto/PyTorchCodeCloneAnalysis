from typing import Tuple, List

import numpy as np
import torch
from numpy import ndarray
from torch.utils.data import DataLoader
from torch import Tensor
import torch.nn as nn
from torchvision import datasets
from fashion_dataset import FashionDataset


class ModelWrapper:
    def __init__(self, model: nn.Module, epochs: int, batch_size: int, model_name:str):
        self.model = model
        datasets.FashionMNIST('./data', )
        self.epochs = epochs
        self.optimizer = model.optimizer
        self.loss_function = model.loss_function
        self.batch_size = batch_size
        self.model_name = model_name

    def train_model(self, training_data, batch_size: int = None) -> Tuple[List[float], List[float]]:
        if batch_size is None:
            batch_size = self.batch_size
        train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
        if batch_size is not None:
            self.model.train()
        loss_list = []
        accuracy_list = []
        for i in range(self.epochs):
            running_loss = 0.0
            total = 0
            correct = 0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels.type(torch.long))
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                predictions = torch.argmax(outputs.data, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
            accuracy_list.append(100 * correct / total)
            loss_list.append(running_loss / total)
            print(f'Epoch: {i + 1} loss: {loss_list[-1]} accuracy: {accuracy_list[-1]}%')
        return loss_list, accuracy_list

    def train(self, training_data) -> Tuple[ndarray, ndarray]:
        loss, accuracy = self.train_model(training_data)
        # for i in range(3):
        #     if accuracy[-1] >= 88:
        #         break
        #     temp_loss, temp_accuracy = self.train_model(training_data, 5)
        #     loss.extend(temp_loss)
        #     accuracy.extend(temp_accuracy)
        return np.array(loss), np.array(accuracy)

    def test(self, data) -> Tuple[float, float]:
        test_loader = DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=2)
        total = 0
        correct = 0
        loss = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                loss += self.loss_function(outputs, labels.type(torch.long)).sum().item()
                predictions = torch.argmax(outputs.data, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        return loss / total, 100 * correct / total

    def predict(self, data: FashionDataset) -> Tensor:
        pred_loader = DataLoader(data, batch_size=data.shape(0), shuffle=False, num_workers=2)
        self.model.eval()
        with torch.no_grad():
            for inputs, _ in pred_loader:
                outputs = self.model(inputs)
                predictions = torch.argmax(outputs.data, 1)
                return predictions

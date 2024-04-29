#!/usr/bin/python3
# -*- coding: utf8 -*-

import torch
import sys
import os
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from net import SimpleConv3Net
from dataset import ImageData
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter


class TrainModel():
    """Docstring for TrainModel. """
    def __init__(self, model, criterion, optimizer, scheduler, num_epochs,
                 data_dir):
        """TODO: to be defined.

        :model: PyTorch model
        :criterion: Loss function
        :optimizer: Optimizer
        :scheduler: lr_scheduler
        :num_epochs: Training epochs
        :data_dir: Data root directory

        """

        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._num_epochs = num_epochs
        self._data_dir = data_dir
        self._train_dir = os.path.join(self._data_dir, 'train')
        self._val_dir = os.path.join(self._data_dir, 'val')
        self._use_gpu = torch.cuda.is_available()
        self.data_transform()
        self._dataset = self.make_datasets()
        self._dataloader = self.make_dataloader()
        self._dataset_size = self.make_dataset_size()

        for epoch in range(self._num_epochs):
            print("Epoch: {}/{}".format(epoch, self._num_epochs))
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train(True)
                else:
                    model.train(False)

                running_loss = 0.0
                running_acc = 0.0

                for data in self._dataloader[phase]:
                    inputs, labels = data
                    if self._use_gpu:
                        model.cuda()
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs = Variable(inputs)
                        labels = Variable(labels)

                    outputs = self._model(inputs)
                    self._optimizer.zero_grad()
                    loss = self._criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        self._optimizer.step()
                        self._scheduler.step()

                    _, preds = torch.max(outputs.data, 1)
                    running_loss += loss.data.item()
                    running_acc += torch.sum(preds == labels).item()

                epoch_loss = running_loss / self._dataset_size[phase]
                epoch_acc = running_acc / self._dataset_size[phase]

                self._writer = SummaryWriter()
                if phase == 'train':
                    self._writer.add_scalar('data/trainloss', epoch_loss,
                                            epoch)
                    self._writer.add_scalar('data/trainacc', epoch_acc, epoch)

                else:
                    self._writer.add_scalar('data/valloss', epoch_loss, epoch)
                    self._writer.add_scalar('data/valacc', epoch_acc, epoch)

                print("{} Loss: {:.4f}, Acc: {:.4%}".format(
                    phase, epoch_loss, epoch_acc))

        self._writer.export_scalars_to_json('./all_scalars.json')
        self._writer.close()

        torch.save(self._model.state_dict(), './models/model.ckpt')

    def data_transform(self):
        self._data_transform = {
            'train':
            transforms.Compose([
                transforms.Resize((150, 150)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            'val':
            transforms.Compose([
                transforms.Resize((150, 150)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        }

    def make_datasets(self):
        dataset = {
            x: ImageData(os.path.join(self._data_dir, x),
                         self._data_transform[x])
            for x in ['train', 'val']
        }

        return dataset

    def make_dataloader(self):
        dataloader = {
            x: DataLoader(self._dataset[x],
                          batch_size=64,
                          shuffle=True,
                          num_workers=4)
            for x in ['train', 'val']
        }

        return dataloader

    def make_dataset_size(self):
        dataset_size = {x: len(self._dataset[x]) for x in ['train', 'val']}

        return dataset_size


if __name__ == "__main__":
    my_model = SimpleConv3Net()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(my_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=100,
                                                gamma=0.1)

    TrainModel(my_model, criterion, optimizer, scheduler, 300, './data/')

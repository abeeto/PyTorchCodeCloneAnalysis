#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import pathlib

import torch
import torchvision
import tqdm
from torch.utils.tensorboard import SummaryWriter

from pytorch_bayes import nn


class Flatten():

    def __call__(self, pic):
        return torch.flatten(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MNISTDataModule(object):

    def __init__(
        self,
        root='data',
        batch_size=128,
    ):
        self.root = pathlib.Path(root)
        # setup dataset
        self.train_dataset, self.test_dataset = self._setup_dataset()
        # configure dataloader
        self.batch_size = batch_size

    def _setup_dataset(self):
        # for fit stage
        train_dataset = torchvision.datasets.MNIST(
            root=self.root, train=True, download=True, transform=self._transform()
        )
        # for predict stage
        test_dataset = torchvision.datasets.MNIST(
            root=self.root, train=False, download=True, transform=self._transform()
        )
        return train_dataset, test_dataset

    def _transform(self):
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            Flatten(),
        ])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )


@nn.utils.variational_approximator
class MNISTModule(nn.BayesianModule):

    def __init__(
        self,
        hidden_features=(400, 400),
        scale_mixture=True,
        sigma_1=math.exp(-0.0),
        sigma_2=math.exp(-6.0),
        pi=0.5,
        mu=0.0,
        sigma=1.0,
        learning_rate=1e-3,
    ):
        super().__init__()
        self.net = nn.BayesianMLP(
            in_features=28 * 28,
            hidden_features=hidden_features,
            out_features=10,
            scale_mixture=scale_mixture,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
            pi=pi,
            mu=mu,
            sigma=sigma,
        )
        self.learning_rate = learning_rate

    def forward(self, input):
        self._log_prior_reset()
        self._log_variational_posterior_reset()
        output = self.net(input)
        self._log_prior = self.net.log_prior
        self._log_variational_posterior = self.net.log_variational_posterior
        return output

    def configure_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def configure_criterion(self):
        return torch.nn.CrossEntropyLoss(reduction='sum')


class BayesByBackprop(object):

    def __init__(
        self,
        datamodule,
        module,
        num_epochs=25,
        num_mc_samples=5,
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_dataloader = datamodule.train_dataloader()
        self.test_dataloader = datamodule.test_dataloader()
        self.net = module.to(self.device)
        self.optimizer = module.configure_optimizer()
        self.criterion = module.configure_criterion().to(self.device)
        self.num_epochs = num_epochs
        self.num_mc_samples = num_mc_samples
        self.progress_bar = tqdm.trange(num_epochs)
        self.writter = SummaryWriter()

    def _training_epoch(self):
        loss_aggregator = 0.0
        self.net.train()
        num_batches = len(self.train_dataloader)
        for batch_idx, batch in enumerate(self.train_dataloader):
            input, target = batch
            input, target = input.to(self.device), target.to(self.device)
            complexity_weight = nn.utils.minibatch_weight(batch_idx, num_batches)
            loss = self.net.mc_elbo(
                self.criterion, input, target,
                complexity_weight,
                self.num_mc_samples,
            )
            loss_aggregator += loss
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            if batch_idx % 100 == 0:
                output = self.net(input)
                acc = torch.eq(output.argmax(dim=1), target).type(torch.float).mean()
                self.progress_bar.set_postfix(loss=loss.item(), acc=acc.item())
        return loss_aggregator / num_batches

    def _validation_epoch(self):
        acc_aggregator = 0.0
        self.net.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_dataloader):
                input, target = batch
                input, target = input.to(self.device), target.to(self.device)
                transformer = torch.nn.Softmax(dim=1).to(self.device)
                pred = self.net.mc_pred(
                    transformer, input,
                    self.num_mc_samples,
                )
                acc_aggregator += torch.eq(pred.argmax(dim=1), target).type(torch.float).sum().item()
        return acc_aggregator / len(dataloader.dataset)

    def fit(self):
        for epoch in self.progress_bar:
            loss = self._training_epoch()
            self.writer.add_scalar('loss/train', loss, epoch)
            acc = self._validation_epoch()
            self.writer.add_scalar('acc/val', acc, epoch)


if __name__ == '__main__':
    NUM_EPOCHS = 20
    # nn
    HIDDEN_FEATURES = (100, 100)
    # bayes by backprop
    NUM_MC_SAMPLES = 5
    SCALE_MIXTURE = True
    SIGMA_1 = math.exp(-0.0)
    SIGMA_2 = math.exp(-6.0)
    PI = 0.5
    MU = 0.0
    SIGMA = 1.0
    # minibatch
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3

    mnist = MNISTDataModule(batch_size=BATCH_SIZE)
    bnn = MNISTModule(
        hidden_features=(400, 400),
        scale_mixture=True,
        sigma_1=math.exp(-0.0),
        sigma_2=math.exp(-6.0),
        pi=0.5,
        mu=0.0,
        sigma=1.0,
        learning_rate=1e-3,
    )
    trainer = BayesByBackprop(
        mnist,
        bnn,
        num_epochs=NUM_EPOCHS,
        num_mc_samples=NUM_MC_SAMPLES,
    )
    trainer.fit()

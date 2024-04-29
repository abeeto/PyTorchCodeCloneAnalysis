"""Trains a simple mnist model."""

import typing
import torch
import torchvision
import torch.nn as nn
import numpy as np
import ignite
import ignite.engine as engine
import ignite.contrib.handlers.param_scheduler as param_scheduler
import helper
from torchluent import FluentModule
import functools
import os
import matplotlib.pyplot as plt
import scipy.signal
import json

class State(typing.NamedTuple):
    """Contains all the state which will be passed around. This would be
    better as a dataclass, but python 3.6 support is still necessary"""
    train_set: torch.utils.data.Dataset
    val_set: torch.utils.data.Dataset
    model: nn.Module
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    optimizer: typing.Callable
    cycle_time: int
    lr_scheduler: param_scheduler.ParamScheduler
    loss: typing.Callable
    evaluator: engine.Engine

    train_metric_samples: int
    val_metric_samples: int

    train_store_metric_samples: int

    stored_ind: typing.Optional[typing.List[int]]
    lrs: typing.Optional[np.ndarray]
    accs: typing.Optional[np.ndarray]

def log_training_results(state: State, tnr: engine.Engine):
    state.evaluator.run(helper.create_loader_for_subset(
        state.train_set, state.train_metric_samples, batch_size=64))
    metrics = state.evaluator.state.metrics

    acc = metrics['accuracy']
    loss = metrics['loss']

    print(f'Training Results - Epoch: {tnr.state.epoch}, acc: {acc:.3f}, '
          + f'loss: {loss:.3f}')

def log_validation_results(state: State, tnr: engine.Engine):
    state.evaluator.run(helper.create_loader_for_subset(
        state.val_set, state.val_metric_samples, batch_size=64
    ))
    metrics = state.evaluator.state.metrics
    acc = metrics['accuracy']
    loss = metrics['loss']
    print(f'Validation Results - Epoch: {tnr.state.epoch}, acc: {acc:.3f}, '
          + f'loss: {loss:.3f}')

def store_lr_vs_acc(state: State, tnr: engine.Engine):
    state.evaluator.run(helper.create_loader_for_subset(
        state.train_set, state.train_store_metric_samples, batch_size=64
    ))
    metrics = state.evaluator.state.metrics
    acc = metrics['accuracy']
    lr = state.lr_scheduler.get_param()
    ind = state.stored_ind[0]
    state.lrs[ind] = lr
    state.accs[ind] = acc
    state.stored_ind[0] = ind + 1

transform = torchvision.transforms.ToTensor()

def train(lr_sweep=False):
    train_set = torchvision.datasets.MNIST('datasets/mnist', download=True, transform=transform)
    val_set = torchvision.datasets.MNIST('datasets/mnist', train=False, download=True,
                                         transform=transform)
    unstripped, model = (
        FluentModule((1, 28, 28,))
        .verbose()
        .wrap(True)
        .conv2d(32, 5)
        .maxpool2d(3)
        .operator('LeakyReLU')
        .save_state()
        .flatten()
        .dense(64)
        .operator('Tanh')
        .save_state()
        .dense(10)
        .save_state()
        .build(with_stripped=True)
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.02) # lr value irrelevant here
    cycle_time_epochs = 6
    cycle_time = len(train_loader) * cycle_time_epochs
    scheduler = param_scheduler.LinearCyclicalScheduler(optimizer, 'lr', 0.001, 0.03, cycle_time)

    loss = torch.nn.CrossEntropyLoss()
    trainer = engine.create_supervised_trainer(model, optimizer, loss)
    evaluator = engine.create_supervised_evaluator(
        model,
        metrics={
            'accuracy': ignite.metrics.Accuracy(),
            'loss': ignite.metrics.Loss(loss)
        }
    )

    state = State(
        train_set=train_set, val_set=val_set, model=model,
        train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, cycle_time=cycle_time, lr_scheduler=scheduler,
        loss=loss, evaluator=evaluator,
        train_metric_samples=128,
        val_metric_samples=64*4,
        train_store_metric_samples=64,
        stored_ind=[0] if lr_sweep else None,
        lrs=np.zeros(cycle_time // 2) if lr_sweep else None,
        accs=np.zeros(cycle_time // 2) if lr_sweep else None
    )

    trainer.add_event_handler(engine.Events.ITERATION_STARTED, scheduler)
    trainer.add_event_handler(
        engine.Events.EPOCH_COMPLETED,
        functools.partial(log_training_results, state))
    trainer.add_event_handler(
        engine.Events.EPOCH_COMPLETED,
        functools.partial(log_validation_results, state))

    if lr_sweep:
        trainer.add_event_handler(
            engine.Events.ITERATION_COMPLETED,
            functools.partial(store_lr_vs_acc, state))

    trainer.run(train_loader, max_epochs=(cycle_time_epochs // 2
                                          if lr_sweep else cycle_time_epochs * 4))

    evaluator.run(train_loader)
    metrics = evaluator.state.metrics # pylint: disable=no-member
    train_acc = metrics['accuracy']
    train_loss = metrics['loss']
    print(f'Training Results - Final: acc: {train_acc:.3f}, loss: {train_loss:.3f}')

    evaluator.run(val_loader)
    metrics = evaluator.state.metrics # pylint: disable=no-member
    val_acc = metrics['accuracy']
    val_loss = metrics['loss']
    print(f'Validation Results - Final: acc: {val_acc:.3f}, loss: {val_loss:.3f}')

    os.makedirs('out/mnist', exist_ok=True)

    torch.save(unstripped, 'out/mnist/model.pt')
    with open('out/mnist/final.json', 'w') as outfile:
        json.dump({
            'train': {'acc': train_acc, 'loss': train_loss},
            'val': {'acc': val_acc, 'loss': val_loss}
        }, outfile)

    if lr_sweep:
        np.savez_compressed('out/mnist/sweep.npz', lrs=state.lrs, accs=state.accs)

        fig, ax = plt.subplots()
        ax.plot(state.lrs, state.accs)
        fig.savefig('out/mnist/out.png')
        plt.close(fig)

        accs_smoothed = scipy.signal.savgol_filter(state.accs, 51, 3)
        fig, ax = plt.subplots()
        ax.plot(state.lrs, accs_smoothed)
        fig.savefig('out/mnist/smoothed.png')
        ax.grid(True)
        fig.savefig('out/mnist/smoothed_grid.png')
        plt.close(fig)

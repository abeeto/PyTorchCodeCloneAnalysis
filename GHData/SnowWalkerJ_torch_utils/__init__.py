from collections import defaultdict
from contextlib import contextmanager
from inspect import isfunction
import numpy as np
import pandas as pd
import tensorboardX as tb
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from progressbar import ProgressBar, Bar, ETA, Percentage
from .core import USE_CUDA, Variable
from .modules import *
from .misc import *


def get_parameter(module, path):
    path = path.split(".")
    for subpath in path:
        module = getattr(module, subpath)
    return module


class Trainer:
    def __init__(self, module: nn.Module, loss_fn, optimizer: optim.Optimizer, logdir: str=None):
        self.module = module
        if USE_CUDA:
            module.cuda()
        self.loss_fn = loss_fn
        self.metrics = {}
        self.add_metric("Loss", loss_fn)
        self.optimizer = optimizer
        self.loss_parameters = []
        self.loss = defaultdict(list)
        self.module.register_forward_pre_hook(lambda m, n: self.clear_cache()) # Clear cache every time the module is called
        self.parameter_watcher = []
        self.output_watcher = []
        self.writer = tb.SummaryWriter(logdir)
        self.batch_size = 32
        self.num_workers = 0

    def clear_cache(self):
        self.loss.clear()

    def train_once(self, dataloader: DataLoader):
        self.module.train()
        for x, y in dataloader:
            x, y = Variable(x), Variable(y)
            self.clear_cache()
            loss = self.get_loss(x, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, dataloader: DataLoader, num_epochs: int, validate_data=None):
        self.batch_size = dataloader.batch_size // 2
        self.num_workers = dataloader.num_workers // 2
        progressbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()])
        for epoch in progressbar(range(num_epochs)):
            self.train_once(dataloader)
            if validate_data is not None:
                if isinstance(validate_data, dict):
                    for name, dataset in validate_data.items():
                        self.show_metrics(epoch, dataset, name)
                else:
                    self.show_metrics(epoch, validate_data)

    def show_metrics(self, epoch: int, dataset: Dataset, dataset_name: str=None):
        self.module.eval()
        dataloader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=USE_CUDA, num_workers=self.num_workers)
        data = defaultdict(list)
        
        for i, (x, y) in enumerate(dataloader):
            x = Variable(x, volatile=True)
            y = Variable(y, volatile=True)
            if i == 0:
                with self.watch(epoch, dataset_name):
                    y_ = self.module(x)
            else:
                y_ = self.module(x)
            for name, func in self.metrics.items():
                if dataset_name is not None:
                    name = "_".join([dataset_name, name])
                value = func(y_, y)
                data[name].append(value.data[0])
        for key, value in data.items():
            self.writer.add_scalar(key, np.mean(value), epoch)

    def save(self, filename):
        torch.save(self.module.state_dict(), filename)

    def add_parameter_loss(self, parameter: nn.Parameter, func):
        """
        Add a loss function that will be applied to a specific parameter.

        Parameters
        ----------
        parameter: nn.Parameter
            the parameter the the loss will be applied to
        func: Function
            the loss function that maps the parameter to a scalar variable,
            such as L1, L2
        """
        self.loss_parameters.append((parameter, func))

    def add_output_loss(self, module: nn.Module, func):
        """
        Add a loss function that will be applied to the output of a module.
        This is usually used to apply restrictions to the distributions of the output.

        Parameters
        ----------
        module: nn.Module
            the module whose output will be set as loss
        func: Function
            the loss function that maps the output to a scalar variable
        """
        if module not in self.module.modules():
            raise RuntimeError("The module of an output loss must be a children of the root module")

        def hook(module, input, output):
            self.loss[module].append(func(output))

        handler = module.register_forward_hook(hook)
        return handler

    def get_loss(self, x, y):
        """
        Total Loss = Parameter Loss + Output Loss + Loss_fn
        """
        loss = self.loss_fn(self.module(x), y)
        output_loss = sum(torch.stack(v).mean() for v in self.loss.values())
        parameter_loss = sum(func(parameter) for parameter, func in self.loss_parameters)
        return loss + output_loss + parameter_loss

    def watch_parameter(self, parameter, name: str, func='hist'):
        """
        Log the distribution of a specific parameter in the module

        Parameters
        ==========
        parameter: nn.Parameter
            The parameter to log
        name: str
            identifier of the watch
        func: str
            One of {'hist', 'mean', 'std'} or a function. If 'hist' and histogram of the parameter
            is logged, otherwise a scalar of the transformed value if logged.
        """
        self.parameter_watcher.append((func, name, parameter))

    def watch_output(self, module, name: str, func='hist'):
        """
        Log the distribution of the output in the module

        Parameters
        ==========
        module: nn.Module
            The output of which to be watched
        name: str
            identifier of the watch
        func: str
            One of {'hist', 'mean', 'std'} or a function. If 'hist' and histogram of the parameter
            is logged, otherwise a scalar of the transformed value if logged.
        """
        if module not in self.module.modules() and module is not self.module:
            raise RuntimeError("The module of an output watcher must be a children"
                               " of the root module or the root module itself")
        self.output_watcher.append((func, name, module))

    @contextmanager
    def watch(self, epoch: int, dataset_name: str=None):
        def hook_factory(name, func, epoch):
            def hook(module, input, output):
                if func == "hist":
                    self.writer.add_histogram(name, output.data.cpu().numpy(), epoch, bins='auto')
                else:
                    if isfunction(func):
                        value = func(output.data)
                    else:
                        value = getattr(output.data, func)()
                    if isinstance(value, (np.ndarray, pd.Series)):
                        self.writer.add_scalars(name, pd.Series(value).to_dict(), epoch)
                    else:
                        self.writer.add_scalar(name, value, epoch)
                
            return hook

        hooks = []
        for func, name, module in self.output_watcher:
            if dataset_name is not None:
                name = "_".join([dataset_name, name])
            handler = module.register_forward_hook(hook_factory(name, func, epoch))
            hooks.append(handler)
        
        yield

        for func, name, parameter in self.parameter_watcher:
            if dataset_name is not None:
                name = "_".join([dataset_name, name])
            if func != "hist" or isfunction(func):
                if isfunction(func):
                    value = func(parameter.data)
                else:
                    value = getattr(parameter.data, func)()
                if isinstance(value, (np.ndarray, pd.Series, list, tuple)):
                    self.writer.add_scalars(name, pd.Series(value).to_dict(), epoch)
                else:
                    self.writer.add_scalar(name, value, epoch)
            elif func == "hist":
                self.writer.add_histogram(name, parameter.data.cpu().numpy(), epoch, bins='auto')

        for hook in hooks:
            hook.remove()

    def add_metric(self, name, func):
        """
        Add a scalar metric regarding the output of the module and the target

        Parameters
        ----------
        name: str
            the name of the metric
        func
            (output, target) -> Scalar Variable
        """
        self.metrics[name] = func

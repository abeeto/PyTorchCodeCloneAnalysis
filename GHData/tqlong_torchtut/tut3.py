import math

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from easydict import EasyDict

from tut3_model import RNN
from trainer import train_model

def get_transform():
    '''
    return: a torch's transform for RGB images, converting them to tensor [0,1]
    '''
    # Image preprocessing modules
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(28),
        transforms.ToTensor()])
    return transform

def get_datasets():
    '''
    return: 
      datasets.train: training set with tranform
      datasets.test: testing set with ToTensor transform only
    '''
    transform = get_transform()
    
    # MNIST dataset
    datasets = EasyDict(dict(
        train=torchvision.datasets.MNIST(root='data/', train=True, transform=transform, download=True),
        test =torchvision.datasets.MNIST(root='data/', train=False, transform=transforms.ToTensor()),
    ))
    return datasets

def get_quick_optimizer(model, max_iter, base_lr=0.001, power=0.9):
    '''
    model: a torch model
    max_iter: maximum number of training steps
    base_lr: base learning rate
    power: power number for learning rate update
    return: tuple (optimizer, scheduler)
      optimizer: Adam with specified learning rate
      scheduler: LambdaLR scheduler with formula $$lr = (1 - iter/max_iter)**power$$
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    lr_update = lambda iter: (1 - iter/max_iter)**power
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_update)
    return optimizer, scheduler

def get_model(config):
    '''
    device: compute device "cpu", "cuda", "cuda:0", "cuda:1"
    return: a torch model on device
    '''
    return RNN(config.input_size, config.hidden_size, config.num_layers, config.num_classes).to(config.device)

def get_loss():
    '''
    return: a loss function
    '''
    return nn.CrossEntropyLoss()

def get_max_iter(train_config, ndata):
    '''
    return: maximum number of training steps $$max_iter = epochs * \lceil n / batch_size\rceil$$
    '''
    return train_config.num_epochs*math.ceil(ndata/train_config.batch_size)

def additional_preprocess(train_config):
    def preprocess(images, labels):
        return images.reshape(-1, train_config.sequence_length, train_config.input_size), labels
    return preprocess

def main():
    '''
    main training code
    '''
    # training setting
    train_config = EasyDict(dict(
        # device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        # RNN parameters
        sequence_length = 28,
        input_size = 28,
        hidden_size = 128,
        num_layers = 2,
        # training configuration
        num_epochs = 10,
        num_classes = 10,
        batch_size = 100,
        num_workers = 6,
        learning_rate = 1e-3,
        power = 0.9,
        model_path = 'runs/rnn_exp_1',
    ))

    # additional objects (model, datasets, criterion, optimizer, scheduler) for training
    train_config.additional_preprocess = additional_preprocess(train_config)
    train_config.datasets  = get_datasets()
    train_config.model     = get_model(train_config)
    train_config.criterion = get_loss()
    max_iter               = get_max_iter(train_config, len(train_config.datasets.train))
    train_config.optimizer, train_config.scheduler\
                           = get_quick_optimizer(train_config.model, max_iter, 
                                                 base_lr=train_config.learning_rate, power=train_config.power)

    train_model(train_config)
        
if __name__ == "__main__":
    main()

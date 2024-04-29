import math

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from easydict import EasyDict

from tut4_model import BiRNN
from trainer import train_model
from tut3 import get_transform, get_datasets, get_quick_optimizer, get_loss, get_max_iter, additional_preprocess

def get_model(config):
    '''
    device: compute device "cpu", "cuda", "cuda:0", "cuda:1"
    return: a torch model on device
    '''
    return BiRNN(config.input_size, config.hidden_size, config.num_layers, config.num_classes).to(config.device)

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
        model_path = 'runs/birnn_exp_1',
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

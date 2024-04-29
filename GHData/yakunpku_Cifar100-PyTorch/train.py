import os
import os.path as osp
import random
import time
import numpy as np
import logging
import shutil
import torch
import torch.nn as nn
from config import setup_logger
from config import Config as cfg
from datasets import create_dataloader
from models import define_net
from losses import losses
from utils.lr_schedulers import WarmUpMultiStepLR, WarmUpCosineLR
from trainer import trainer

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="The arguments for training the classifier on CIFAR-100 dataset.")

    ## Network Architecture
    parser.add_argument('--arch', type=str,
                        required=True, 
                        choices=['resnet-20', 'resnet-56', 'resnet-110'],
                        help="the architecture of classifier network, which is only support : [resnet-20, resnet-56, resnet-110] currently!")
    parser.add_argument('--block-name', type=str, default='BasicBlock',
                        help="the building block for resnet : BasicBlock, Bottleneck (default: Basicblock for CIFAR-10/CIFAR-100)")

    parser.add_argument('--num-classes', type=int,
                        default=100,
                        help="the number of classes in the classification dataset")

    ## Loss Function
    parser.add_argument('--loss_type', type=str,
                        default='ce',
                        choices=['ce', 'ls_ce'],
                        help="the loss function for network: ce: Cross Entropy; ls_ce: Label Smoothing Cross Entropy")
    
    ## Optimization
    parser.add_argument('--optimizer', type=str,
                        default='SGD',
                        choices=['SGD', 'Adam'],
                        help="the optimizer for training networks")
    parser.add_argument('--momentum', type=float,
                        default=0.9,
                        help="the momentum for optimizer")
    parser.add_argument('--learning-rate', '-lr', dest='lr', 
                        type=float, 
                        default=1.0e-2, 
                        help="the learning rate for training networks")
    parser.add_argument('--lr-gamma',
                        type=float, 
                        default=0.1,
                        help="the learning rate decay coefficient")
    parser.add_argument('--weight-decay', '-wd', dest='wd', 
                        type=float, 
                        default=5.0e-4, 
                        help="the weight decay in network")
    parser.add_argument('--scheduler',
                        type=str, 
                        default='step_lr',
                        choices=['step_lr', 'cosine_lr'], 
                        help="the lr scheduler")
    parser.add_argument('--warmup-method',
                        type=str, 
                        default='linear',
                        choices=['constant', 'linear'], 
                        help="the lr warmup method")
    parser.add_argument('--warmup-step',
                        type=int,
                        default=5,
                        help="the lr warmup step")
    parser.add_argument('--milestones',
                        nargs='+',
                        type=int,
                        default=[60, 120, 160],
                        help="milestones for the learning rate decay")

    ## Train Configs
    parser.add_argument('--pretrained', action='store_true',
                        help="To initialized the network weights with pretrained weights on ImageNet, which is not support on resnet cifar model.")
    parser.add_argument('--num-epochs', type=int, 
                        default=200, 
                        help="the max number of epochs") 
    parser.add_argument('--batch-size', type=int, 
                        default=128, 
                        help="the dataset batch size for training network")
    parser.add_argument('--num-workers', type=int,
                        default=6,
                        help="the number of workers for loading dataset")
    parser.add_argument('--seed', type=int,
                        default=10007,
                        help="the random seed")

    ## Devices
    parser.add_argument('--gpu', type=int, 
                        default=0, 
                        help="to assign the gpu device to train the network") 

    ## Checkpoints
    parser.add_argument('--checkpoint-cycle', type=int, 
                        default=5, 
                        help="the cycle epoch to store checkpoint")
    parser.add_argument('--resume', default='', type=str, 
                    help="path to previous checkpoint (default: '')")
    parser.add_argument('--model-store-dir', type=str, 
                        default='./experiments', 
                        help="the classification model store folder")
    args = parser.parse_args()
    return args

class Runner(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info(self.args)

    def build_optimizer(self, network):
        optim_params = []
        for k, v in network.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
        
        if self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(optim_params, lr=self.args.lr, weight_decay=self.args.wd, momentum=self.args.momentum)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(optim_params, lr=self.args.lr, weight_decay=self.args.wd)
        else:
            raise NotImplementedError('The optimizer: {} is not implemented.'.format(self.args.optimizer))
        
        return optimizer

    def build_loss_func(self):
        if self.args.loss_type == 'ce':
            loss_func = nn.CrossEntropyLoss()
        elif self.args.loss_type == 'ls_ce':
            loss_func = losses.LabelSmoothingCrossEntropy()
        else:
            raise NotImplementedError('The loss function: {} is not implemented.'.format(self.args.loss_type))
        
        return loss_func

    def build_scheduler(self, optimizer, start_epoch=0):
        if self.args.scheduler == 'step_lr':
            lr_scheduler = WarmUpMultiStepLR(optimizer, self.args.milestones, gamma=self.args.lr_gamma, warmup_factor=1.0e-4, 
                                                warmup_iters=self.args.warmup_step, warmup_method=self.args.warmup_method, last_epoch=start_epoch-1)
        elif self.args.scheduler == 'cosine_lr':
            lr_scheduler = WarmUpCosineLR(optimizer, self.args.num_epochs, warmup_factor=1.0e-4,
                                                warmup_iters=self.args.warmup_step, warmup_method=self.args.warmup_method, last_epoch=start_epoch-1)
        else:
            raise NotImplementedError('The learning rate scheduler {} is not implemented.'.format(self.args.scheduler))
        return lr_scheduler

    def run(self):
        device = 'cuda:{}'.format(self.args.gpu)

        train_dataloader = create_dataloader(cfg.train_image_dir, cfg.train_image_list, phase='train', 
            batch_size=self.args.batch_size, num_workers=self.args.num_workers)
        test_dataloader = create_dataloader(cfg.test_image_dir, cfg.test_image_list, phase='test')

        model_store_path = osp.join(self.args.model_store_dir, self.args.arch)
        
        if osp.exists(model_store_path):
            shutil.rmtree(model_store_path)
        os.makedirs(model_store_path, exist_ok=True)

        network = define_net(self.args.arch, self.args.block_name, self.args.num_classes, pretrained=self.args.pretrained).to(device)

        optimizer = self.build_optimizer(network)
        
        start_epoch = 0
        if self.args.resume:
            checkpoint = torch.load(self.args.resume)
            start_epoch = checkpoint['epoch']
            network.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

        lr_scheduler = self.build_scheduler(optimizer, start_epoch)
        loss_func = self.build_loss_func()

        trainer.Trainer(self.args, 
                        device, 
                        start_epoch,
                        network, 
                        optimizer, 
                        lr_scheduler, 
                        loss_func, 
                        train_dataloader, 
                        test_dataloader, 
                        model_store_path, 
                        self.logger
                        ).train()

def main():
    args = parse_args()

    set_random_seed(args.seed if args.seed != -1 else (int(round(time.time() * 1000)) % (2**32 -1)))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    setup_logger('base', 'train')
    logger = logging.getLogger('base')

    Runner(args, logger).run()

if __name__ == '__main__':
    main()

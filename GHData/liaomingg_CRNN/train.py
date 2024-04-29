# -*- coding: utf-8 -*-
# train.py
# author: lm


import argparse
import json
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as distributed
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import yaml
from easydict import EasyDict as edict
from torch import optim
from torch.utils.data.dataloader import DataLoader

from data.crnn_transform import CRNNCollateFN
from data.label_converter import CTCLabelConverter
from data.lmdb_dataset import LMDBDataSet
from metrics.rec_metric import RecMetric
from model.crnn import CRNN
from scripts.lr_schedule import WarmupCosineAnnealingLR
from scripts.utils import (AverageMeter, ProgressMeter, load_checkpoint,
                           save_checkpoint)

try:
    from warpctc_pytorch import CTCLoss
except:
    from torch.nn import CTCLoss



os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'

def sgd_optimizer(model:nn.Module, lr:float, momentum:float, weight_decay:float):
    """[summary]

    param: model (nn.Module): 
    param: lr (float): 
    param: momentum (float): 
    param: weight_decay (float): 

    Returns:
        [type]: [description]
    """
    params = []
    for k, v in model.named_parameters():
        if not v.requires_grad:
            continue
        apply_weight_decay = weight_decay
        apply_lr = lr 
        if 'bias' in k or 'bn' in k:
            apply_weight_decay = 0
            print('set weight decay = 0 for {}'.format(k))
        if 'bias' in k:
            apply_lr = 2 * lr # caffe 
        params += [{'params': [v], 'lr': apply_lr, 'weight_decay': apply_weight_decay}]
    optimizer = torch.optim.SGD(params, lr, momentum=momentum)
    return optimizer




def main(args):
    # check cpu or gpu!
    if args.gpu is not None and torch.cuda.is_available():
        args.gpu = [int(gpu) for gpu in args.gpu.split(',')]
        # args.gpus_per_node = len(args.gpu)
        print('Using GPU: {} for training.'.format(args.gpu))
        print('gpu per node:', args.gpus_per_node)
    else:
        args.gpu = None 
        print('Using CPU for training.')
    
    # distributed training
    if args.distributed:
        args.dataset.train.shuffle = False 
        # default init_method is 'env://'
        if args.dist_url == 'env://' and args.rank == -1:
            args.rank = os.environ['RANK']
            
        if args.multi_processing_distributed:
            args.rank = args.rank * args.gpus_per_node
            
        distributed.init_process_group(backend = args.dist_backend,
                                       init_method = args.dist_url,
                                       world_size = args.world_size,
                                       rank = args.rank)
    
    alphabet = CTCLabelConverter(args.model.alphabet, args.model.ignore_case)
    train_dataset, train_loader = [], [] 
    if args.dataset.train.enable:
        print('Loading train dataset...')
        train_dataset = LMDBDataSet(args.dataset.train.path,
                                    args.dataset.train.shuffle)
        train_sampler = None 
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset,
                                batch_size=args.dataset.train.batch_size,
                                shuffle=args.dataset.train.shuffle,
                                num_workers=args.workers,
                                pin_memory=True,
                                sampler=train_sampler,
                                collate_fn=CRNNCollateFN(args.dataset.height, args.dataset.width))
        print('Traing batchs: {}'.format(len(train_loader)))
    else:
        print('Skip training process.')
        
    # load validate dataset.
    val_dataset, val_loader = [], [] 
    if args.dataset.val.enable:
        print('Loading validate dataset...')
        val_dataset = LMDBDataSet(args.dataset.val.path,
                                args.dataset.val.shuffle)
        
        val_loader = DataLoader(val_dataset,
                                batch_size=args.dataset.val.batch_size,
                                shuffle=args.dataset.val.shuffle,
                                num_workers=args.workers,
                                pin_memory=True,
                                collate_fn=CRNNCollateFN())
        print('Val batchs: {}'.format(len(val_dataset)))
        
    else:
        print('Skip validate during training process.')
    
    print('Train examples: {}, Val examples: {}.'.format(len(train_dataset), len(val_dataset)))
    
    model = CRNN(args.model.in_channels,
                 args.model.hidden_channels,
                 len(alphabet.alphabet) + 1,
                 args.model.act)
    
    if args.model.display:
        print('Model structure.')
        print(model)
    
    best_metric = {
        'acc': 0.0,
        'norm_edit_dist': 0.0}    
    if args.resume:
        # maybe load checkpoint before move to cuda.
        if os.path.isfile(args.resume):
            print(' ===> resume parameters from: {}'.format(args.resume))
            state = load_checkpoint(model, args.resume)
            best_metric = state['metrics']
        else:
            print(' xxx> no checkpoint found at: {}'.format(args.resume))
    else:
        print('training model from scratch.')

    # set GPU or CPU
    if not torch.cuda.is_available() or not args.gpu:
        print('CUDA is unavailable!!! Using CPU training will be slow!!!')
          
    elif args.distributed:
            model = model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, 
                                                              find_unused_parameters=True)
            print('DistributedDataparallel training with selected GPUS: {}'.format(args.gpu))
            
    elif len(args.gpu) <= 1:
        model = model.cuda()
        print('Training model with single GPU: {}'.format(args.gpu))
    else:
        model = torch.nn.DataParallel(model).cuda()
        print('Training model with Data Parallel.')
        
    if CTCLoss is nn.CTCLoss:
        criterion = CTCLoss()
        print('Using torch.nn.CTCLoss()')
    else:
        criterion = CTCLoss(size_average=True)
        print('Using warpctc_pytorch.CTCLoss()')
    if args.gpu:
        criterion = criterion.cuda()

        
    # optimizer
    # optimizer = sgd_optimizer(model, args.lr, args.momentum, args.weight_decay) 
    optimizer = optim.Adam(model.parameters(), args.lr, amsgrad=True)
    # lr_schedule
    lr_scheduler = WarmupCosineAnnealingLR(optimizer, 
                                           args.epochs * len(train_loader) // args.gpus_per_node, 
                                           warmup=args.warmup)
    if args.cudnn_benchmark:
        cudnn.benchmark = True 

    print('Best metric:', best_metric)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and train_sampler:
                train_sampler.set_epoch(epoch)
                
        train(args, model, train_loader, criterion, optimizer, lr_scheduler, alphabet, epoch)
        
        if args.dataset.val.enable and epoch % args.dataset.val.interval == 0:
            metric = val(args, model, val_loader, criterion, alphabet, epoch)
            print('val results:', metric)
        else:
            metric = {'acc': 0., 'norm_edit_dist': 0.}
        
        if not args.multi_processing_distributed or (args.multi_processing_distributed and args.rank % args.gpus_per_node == 0):
            state = {
                'epoch': epoch,
                'name': args.name,
                'state_dict': model.state_dict(),
                'metrics': metric,
                'optimizer': optimizer,
                'scheduler': lr_scheduler.state_dict()}
            save_checkpoint_crnn(state, args, best_metric['acc'] <= metric['acc'])
            if best_metric['acc'] <= metric['acc']:
                best_metric = metric
                
    print('Train complete!')


    
def train(args : edict, 
          model : nn.Module, 
          data_loader : DataLoader,
          criterion : CTCLoss,
          optimizer : optim.Optimizer, 
          lr_scheduler : optim.lr_scheduler._LRScheduler, 
          alphabet: CTCLabelConverter,
          epoch: int):
    ''''''
    batch_time = AverageMeter('BatchTime', ':.3f')
    data_time = AverageMeter('DataTime', ':.3f')
    ctc_time = AverageMeter('CTCTime', ':.3f')
    loss = AverageMeter('Loss', ':.3f')
    acc = AverageMeter('Acc', ':.3f')
    norm_edit_dist = AverageMeter('NormEditDist', ':.3f')
    progress = ProgressMeter(len(data_loader),
        [batch_time, data_time, ctc_time, loss, acc, norm_edit_dist],
        prefix='Epoch: {}'.format(epoch))
    
    metric = RecMetric()
    
    print('Epoch: {}, lr: {:.3e}'.format(epoch, lr_scheduler.get_lr()[0]))
    model.train() # convert to training mode.
    
    st = time.time()
    for idx, data in enumerate(data_loader):
        data_time.update(time.time() - st)
        images, texts = data 
        labels, lenghts = alphabet.encode_batch(texts)
        # to cuda
        if args.gpu:
            images = images.cuda() # only images to cuda, labels no need to cuda.
                    
        preds = model(images) # [b, t, c]
        preds_labels = torch.argmax(preds.detach(), dim=-1).view(-1) # get pred label.
        preds = preds.transpose(1, 0) # [t, b, c]
        if isinstance(criterion, nn.CTCLoss):
            preds = preds.log_softmax(2)
        t, b, _ = preds.size()
        preds_lengths = torch.IntTensor([t] * b)

        ctc_time_start = time.time()
        cost = criterion(preds, labels, preds_lengths, lenghts)
        ctc_time.update(time.time() - ctc_time_start)

        model.zero_grad()
        cost.backward()
        optimizer.step()   
        
        loss.update(cost.item())
        # NOTE: decode by cpu is faster than decode by gpu.
        preds_text = alphabet.decode_batch(preds_labels.cpu(), preds_lengths.cpu())
        
        m = metric(preds_text, texts)
        
        acc.update(m['acc'])
        norm_edit_dist.update(m['norm_edit_dist'])
        
        batch_time.update(time.time() - st)     
        st = time.time()
        lr_scheduler.step()
        if idx % args.display_freq == 0:
            progress.display(idx)

def val(args: edict, 
        model: nn.Module, 
        data_loader: DataLoader, 
        criterion: CTCLoss, 
        alphabet: CTCLabelConverter, 
        epoch: int):
    '''验证模型'''
    batch_time = AverageMeter('BatchTime', ':.3f')
    loss = AverageMeter('Loss', ':.3e')
    img_loss = AverageMeter('ImageLoss', ':.3e')
    word_loss = AverageMeter('WordLoss', ':.3e')
    acc = AverageMeter('Acc', ':.3f')
    norm_edit_dist = AverageMeter('NormEditDist', ':.3f')
    progress = ProgressMeter(len(data_loader),
                             [batch_time, loss, img_loss, word_loss, acc, norm_edit_dist],
                             prefix='Test: ')
    print('validate model...')
    model.eval() # convert to evaluate mode.
    metric = RecMetric()
    
    with torch.no_grad():
        st = time.time()
        for idx, data in enumerate(data_loader):
            images, texts = data 
            labels, lengths = alphabet.encode_batch(texts)
            if args.gpu:
                images = images.cuda()
                
            preds = model(images) # [b, t, c]
            preds_labels = torch.argmax(preds, dim=-1).contiguous().view(-1)
            preds = preds.transpose(1, 0) # [t, b, c]
            if isinstance(criterion, nn.CTCLoss):
                preds = preds.log_softmax(2)
            t, b, _ = preds.size()
            preds_lengths = torch.IntTensor([t] * b)
            cost = criterion(preds, labels, preds_lengths, lengths)
            
            # NOTE: decode by cpu is faster than decode by gpu.
            preds_text = alphabet.decode_batch(preds_labels.cpu(), preds_lengths.cpu())
            m = metric(preds_text, texts) 
            loss.update(cost.item())
            norm_edit_dist.update(m['norm_edit_dist'])
            acc.update(m['acc'])
            
            
            batch_time.update(time.time() - st)
            st = time.time()
            if idx % args.display_freq == 0:
                progress.display(idx)
    return metric.get_metric()
        
    
def save_checkpoint_crnn(state, args, is_best=False, name='ckpt.pth.tar', best_name='best.pth.tar'):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print('Create folder: {}'.format(args.save_dir))
    ckpt_path = os.path.join(args.save_dir, '{}_{}'.format(args.name, name))
    best_path = os.path.join(args.save_dir, '{}_{}'.format(args.name, best_name))
    save_checkpoint(state, ckpt_path, is_best, best_path)



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', default='configs/demo.yaml',
                           help='the path of specified config.')
    args = argparser.parse_args()
    print(args)
    args = edict(yaml.load(open(args.config), Loader=yaml.FullLoader))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
    print(json.dumps(args, indent=2))
    main(args)
    
    
from __future__ import division
import os.path as osp
import sys
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
# from torch.nn.parallel import DistributedDataParallel

from config import config
from dataloader import get_train_loader
from dfn import DFN
from datasets import VOC
from utils.init_func import init_weight, group_weight
from engine.lr_policy import PolyLR
from engine.engine import Engine
from seg_opr.loss_opr import SigmoidFocalLoss
from seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d
from utils import board
import os.path as osp
try:
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")

parser = argparse.ArgumentParser()
'''
PolyLR:
    lr(1-iter/max_iter)^power
'''
'''
VOC:
    from BaseDataset
    all it's data.Dataset struct
    so  we must have __len__ ,__getitem__
    from here we must 4 para
    img_root: here we get the orginal img
    gt_root: here we get groundtruth img
    train_source: file.txt --->using to train
    test_source:file.txt --->using to eval
'''
with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True

    seed = config.seed

    if config.tensorboardX:
        viz = board.Visualizer("./")
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    '''
    # data loader
    1.reading the VOC data, and train_sampler number
    2.according to paper we have flip scale(0.5-1.75) here we change to 0.5-2
    3.get border imformation using canny algorithm
    4.crop the scale img , according the config imformation. normally this part is 512
    '''
    train_loader, train_sampler = get_train_loader(engine, VOC)


    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean',
                                    ignore_index=255)
    aux_criterion = SigmoidFocalLoss(ignore_label=255, gamma=2.0, alpha=0.25)
    if engine.distributed:
        BatchNorm2d = SyncBatchNorm
    else:
        BatchNorm2d= nn.BatchNorm2d

    pretrained_model = osp.abspath(config.pretrained_model)
    model = DFN(config.num_classes, criterion=criterion,
                aux_criterion=aux_criterion, alpha=config.aux_loss_alpha,
                pretrained_model=pretrained_model,
                norm_layer=BatchNorm2d)
    init_weight(model.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr * engine.world_size
    params_list = []

    #here we must know backbone means resnet-101
    params_list = group_weight(params_list, model.backbone,
                               BatchNorm2d, base_lr)
    #business means the smooth net ,and border net
    for module in model.business_layer:
        params_list = group_weight(params_list, module, BatchNorm2d,
                                   base_lr * 10)
    #params_list is list save the weight_group for backbone, smooth net , and border net
    optimizer = torch.optim.SGD(params_list,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = PolyLR(base_lr, config.lr_power, total_iteration)

    if engine.distributed:
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DataParallelModel(model, device_ids=engine.devices)
        model.to(device)

    #register_training_state
    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    model.train()
    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)
        for idx in pbar:
            optimizer.zero_grad()
            #update the iteration imfo
            engine.update_iteration(epoch, idx)

            minibatch = dataloader.next()
            imgs = minibatch['data']
            gts = minibatch['label']
            cgts = minibatch['aux_label']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            cgts = cgts.cuda(non_blocking=True)

            loss = model(imgs, gts, cgts)
            # reduce the whole loss over multi-gpu
            if engine.distributed:
                dist.all_reduce(loss, dist.ReduceOp.SUM)
                loss = loss / engine.world_size
            # else:
            #     loss = Reduce.apply(*loss) / len(loss)

            current_idx = epoch * config.niters_per_epoch + idx
            if(config.tensorboardX):
                viz.line("train/loss",loss.item(),current_idx)
            lr = lr_policy.get_lr(current_idx)

            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10

            loss.backward()
            optimizer.step()
            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.2f' % loss.item()

            pbar.set_description(print_str, refresh=False)

        if (epoch > config.nepochs - 20) or (epoch % config.snapshot_iter == 0):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)

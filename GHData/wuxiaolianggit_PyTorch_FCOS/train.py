from __future__ import division

import os
import argparse
import time
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.voc import VOCDetection
from data.coco import COCODataset
from config.fcos_config import fcos_config
from data.transforms import TrainTransforms, ValTransforms

from utils import distributed_utils
from utils.criterion import build_criterion
from utils.com_flops_params import FLOPs_and_Params
from utils.misc import detection_collate
from utils.vis import vis_data

from evaluator.coco_evaluator import COCOAPIEvaluator
from evaluator.voc_evaluator import VOCAPIEvaluator

from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='FCOS Detection')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--num_gpu', default=1, type=int, 
                        help='Number of GPUs to train')
    parser.add_argument('--eval_epoch', type=int,
                            default=2, help='interval between evaluations')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='Gamma update for SGD')

    # input image size               
    parser.add_argument('--train_min_size', type=int, default=800,
                        help='The shorter train size of the input image')
    parser.add_argument('--train_max_size', type=int, default=1333,
                        help='The longer train size of the input image')
    parser.add_argument('--val_min_size', type=int, default=800,
                        help='The shorter val size of the input image')
    parser.add_argument('--val_max_size', type=int, default=1333,
                        help='The longer val size of the input image')

    # visualize
    parser.add_argument('--vis_data', action='store_true', default=False,
                        help='visualize input data.')
    parser.add_argument('--vis_targets', action='store_true', default=False,
                        help='visualize the targets.')
    parser.add_argument('--vis_anchors', action='store_true', default=False,
                        help='visualize anchor boxes.')
    # model
    parser.add_argument('-m', '--model', default='fcos',
                        help='fcos, fcos_rt')
    parser.add_argument('-mc', '--model_conf', default='fcos_r50_fpn_1x',
                        help='fcos_r50_fpn_1x, fcos_r101_fpn_1x')
    parser.add_argument('--conf_thresh', default=0.05, type=float,
                        help='NMS threshold')
    parser.add_argument('--nms_thresh', default=0.6, type=float,
                        help='NMS threshold')

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc, widerface, crowdhuman')
    
    # Loss
    parser.add_argument('--loss_cls_weight', default=1.0, type=float,
                        help='weight of cls loss')
    parser.add_argument('--loss_reg_weight', default=1.0, type=float,
                        help='weight of reg loss')
    parser.add_argument('--loss_ctn_weight', default=1.0, type=float,
                        help='weight of ctn loss')

    # train trick
    parser.add_argument('--no_warmup', action='store_true', default=False,
                        help='do not use warmup')
    parser.add_argument('--wp_iter', type=int, default=500,
                        help='The upper bound of warm-up')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')
    parser.add_argument('--accumulate', type=int, default=1,
                        help='accumulate gradient')

    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--local_rank', type=int, default=0, 
                        help='local_rank')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # path to save model
    path_to_save = os.path.join(args.save_folder, args.dataset, args.model_conf)
    os.makedirs(path_to_save, exist_ok=True)

    # set distributed
    local_rank = 0
    if args.distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = torch.distributed.get_rank()
        print(local_rank)
        torch.cuda.set_device(local_rank)

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # FCOS-RT Config
    print('Model: ', args.model_conf)
    cfg = fcos_config[args.model_conf]

    # dataset and evaluator
    dataset, evaluator, num_classes = build_dataset(args, cfg, device)

    # dataloader
    dataloader = build_dataloader(args, dataset, detection_collate)

    # criterion
    criterion = build_criterion(args=args, device=device, cfg=cfg, num_classes=num_classes)
    
    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")

    # build model
    net = build_model(args=args, 
                      cfg=cfg,
                      device=device, 
                      num_classes=num_classes, 
                      trainable=True)
    model = net
    model = model.to(device).train()

    # compute FLOPs and Params
    if local_rank == 0:
        model.trainable = False
        model.eval()
        FLOPs_and_Params(model=model, 
                         min_size=args.val_min_size, 
                         max_size=args.val_max_size, 
                         device=device)
        model.trainable = True
        model.train()

    # DDP
    if args.distributed and args.num_gpu > 1:
        print('using DDP ...')
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # SyncBatchNorm
    if args.sybn and args.cuda and args.num_gpu > 1:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # use tfboard
    tblogger = None
    if args.tfboard:
        print('use tensorboard ...')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
        log_path = os.path.join('log/', args.dataset, c_time)
        os.makedirs(log_path, exist_ok=True)

        tblogger = SummaryWriter(log_path)
    
    # optimizer setup
    tmp_lr = base_lr = args.lr
    optimizer = optim.SGD(model.parameters(), 
                            lr=tmp_lr, 
                            momentum=0.9,
                            weight_decay=1e-4)
    # lr schedule
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['lr_epoch'])

    # training configuration
    max_epoch = cfg['max_epoch']
    batch_size = args.batch_size
    epoch_size = len(dataset) // (batch_size * args.num_gpu)
    best_map = -1.
    warmup = not args.no_warmup

    t0 = time.time()
    # start training loop
    for epoch in range(args.start_epoch, max_epoch):
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)            

        # train one epoch
        for iter_i, (images, targets, masks) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size
            # warmup
            if ni < args.wp_iter and warmup:
                nw = args.wp_iter
                tmp_lr = base_lr * pow(ni / nw, 4)
                set_lr(optimizer, tmp_lr)

            elif ni == args.wp_iter and warmup:
                # warmup is over
                warmup = False
                tmp_lr = base_lr
                set_lr(optimizer, tmp_lr)

            # visualize input data
            if args.vis_data:
                vis_data(images, targets, masks)

            # to device
            images = images.to(device)
            masks = masks.to(device)

            # inference
            outputs = model(images, masks=masks)

            # compute loss
            cls_loss, reg_loss, ctn_loss, total_loss = criterion(
                                                        outputs=outputs,
                                                        targets=targets,
                                                        images=images,
                                                        vis_labels=args.vis_targets)
            
            total_loss = total_loss / args.accumulate

            loss_dict = dict(
                cls_loss=cls_loss,
                reg_loss=reg_loss,
                ctn_loss=ctn_loss,
                total_loss=total_loss
            )
            loss_dict_reduced = distributed_utils.reduce_loss_dict(loss_dict)

            # check loss
            if torch.isnan(total_loss):
                continue

            # Backward and Optimize
            total_loss.backward()        
            if ni % args.accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            # display
            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    tblogger.add_scalar('cls loss',  loss_dict_reduced['cls_loss'].item(),  ni)
                    tblogger.add_scalar('reg loss',  loss_dict_reduced['reg_loss'].item(),  ni)
                    tblogger.add_scalar('ctn loss',  loss_dict_reduced['ctn_loss'].item(),  ni)
                
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f][Loss: cls %.2f || reg %.2f || ctn %.2f || size (%d, %d) || time: %.2f]'
                        % (epoch+1, 
                           max_epoch, 
                           iter_i, 
                           epoch_size, 
                           tmp_lr,
                           loss_dict['cls_loss'].item(), 
                           loss_dict['reg_loss'].item(), 
                           loss_dict['ctn_loss'].item(), 
                           images.size(-2), images.size(-1), 
                           t1-t0),
                        flush=True)

                t0 = time.time()

        lr_scheduler.step()

        # evaluation
        if (epoch + 1) % args.eval_epoch == 0 or (epoch + 1) == max_epoch:
            model_eval = model.module if args.distributed else model

            if evaluator is None:
                print('No evaluator ...')
                print('Saving state, epoch:', epoch + 1)
                torch.save(model_eval.state_dict(), os.path.join(path_to_save, 
                            args.model_conf + '_' + repr(epoch + 1) + '.pth'))  
                print('Keep training ...')
            else:
                print('eval ...')

                # set eval mode
                model_eval.trainable = False
                model_eval.eval()

                if local_rank == 0:
                    # evaluate
                    evaluator.evaluate(model_eval)

                    cur_map = evaluator.map
                    if cur_map > best_map:
                        # update best-map
                        best_map = cur_map
                        # save model
                        print('Saving state, epoch:', epoch + 1)
                        torch.save(model_eval.state_dict(), os.path.join(path_to_save, 
                                    args.model_conf + '_' + repr(epoch + 1) + '_' + str(round(best_map*100, 1)) + '.pth'))  
                    if args.tfboard:
                        if args.dataset == 'voc':
                            tblogger.add_scalar('07test/mAP', evaluator.map, epoch)
                        elif args.dataset == 'coco':
                            tblogger.add_scalar('val/AP50_95', evaluator.ap50_95, epoch)
                            tblogger.add_scalar('val/AP50', evaluator.ap50, epoch)

                if args.distributed:
                    # wait for all processes to synchronize
                    dist.barrier()

                # set train mode.
                model_eval.trainable = True
                model_eval.train()
    
    if args.tfboard:
        tblogger.close()


def build_dataset(args, cfg, device):
    if args.dataset == 'voc':
        data_dir = os.path.join(args.root, 'VOCdevkit')
        num_classes = 20
        random_size = cfg["random_size"] if args.multi_scale else None
        dataset = VOCDetection(
                        data_dir=data_dir,
                        transform=TrainTransforms(min_size=args.train_min_size, 
                                                  max_size=args.train_max_size,
                                                  random_size=random_size))

        evaluator = VOCAPIEvaluator(
                        data_dir=data_dir,
                        device=device,
                        transform=ValTransforms(min_size=args.val_min_size, 
                                                max_size=args.val_max_size))

    elif args.dataset == 'coco':
        data_dir = os.path.join(args.root, 'COCO')
        num_classes = 80
        random_size = cfg["random_size"] if args.multi_scale else None
        dataset = COCODataset(
                    data_dir=data_dir,
                    image_set='train2017',
                    transform=TrainTransforms(min_size=args.train_min_size, 
                                              max_size=args.train_max_size,
                                              random_size=random_size))

        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        device=device,
                        transform=ValTransforms(min_size=args.val_min_size, 
                                                max_size=args.val_max_size))
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    return dataset, evaluator, num_classes


def build_dataloader(args, dataset, collate_fn=None):
    # distributed
    if args.distributed and args.num_gpu > 1:
        # dataloader
        dataloader = torch.utils.data.DataLoader(
                        dataset=dataset, 
                        batch_size=args.batch_size, 
                        collate_fn=collate_fn,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        sampler=torch.utils.data.distributed.DistributedSampler(dataset)
                        )

    else:
        # dataloader
        dataloader = torch.utils.data.DataLoader(
                        dataset=dataset, 
                        shuffle=True,
                        batch_size=args.batch_size, 
                        collate_fn=collate_fn,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )
    return dataloader


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()

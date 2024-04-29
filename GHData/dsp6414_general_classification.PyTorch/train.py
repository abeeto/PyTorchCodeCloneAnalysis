# coding : utf-8

import os
import sys
import time
import pyaml
import argparse
import torch.nn as nn
from datetime import datetime

from configs import cfg
from data_loader import Dataset, create_loader
from models import create_model, resume_checkpoint, load_checkpoint, convert_model
from loss import LabelSmoothingCrossEntropy
from optim import create_optimizer
from scheduler import create_scheduler
from utils.torchsummary import summary
from utils.utils import *
from utils.logger import setup_logger


def _parse_args():
    parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('--config_file', default='', type=str, metavar='FILE', required=True,
                        help='path to config file')
    parser.add_argument('--initial_checkpoint', default=None, type=str, metavar='FILE',
                        help='Initialize model from this checkpoint (default: none)')
    parser.add_argument('--resume', default=None, type=str, metavar='FILE',
                        help='Resume full model and optimizer state from checkpoint (default: none)')
    parser.add_argument('--no_resume_opt', action='store_true', default=False,
                        help='prevent resume of optimizer state when resuming model')
    parser.add_argument('--start_epoch', default=None, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--seed', type=int, default=1024, metavar='S',
                        help='random seed (default: 1024)')
    parser.add_argument('--gpu', type=str, default='',
                        help='gpu list to use, for example 1,2,3')

    args = parser.parse_args()

    if len(args.config_file) == 0:
        raise ValueError('Please input config file path!')
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg, args


def main():
    cfg, args = _parse_args()
    torch.manual_seed(args.seed)

    output_base = cfg.OUTPUT_DIR if len(cfg.OUTPUT_DIR) > 0 else './output'
    exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), cfg.MODEL.ARCHITECTURE, str(cfg.INPUT.IMG_SIZE)])
    output_dir = get_outdir(output_base, exp_name)
    with open(os.path.join(output_dir, 'config.yaml'), 'w', encoding='utf-8') as file_writer:
        # cfg.dump(stream=file_writer, default_flow_style=False, indent=2, allow_unicode=True)
        file_writer.write(pyaml.dump(cfg))
    logger = setup_logger(file_name=os.path.join(output_dir, 'train.log'), control_log=False, log_level='INFO')

    # create model
    model = create_model(
        cfg.MODEL.ARCHITECTURE,
        num_classes=cfg.MODEL.NUM_CLASSES,
        pretrained=True,
        in_chans=cfg.INPUT.IN_CHANNELS,
        drop_rate=cfg.MODEL.DROP_RATE,
        drop_connect_rate=cfg.MODEL.DROP_CONNECT,
        global_pool=cfg.MODEL.GLOBAL_POOL)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpu_list = list(map(int, args.gpu.split(',')))
    device = 'cuda'
    if len(gpu_list) == 1:
        model.cuda()
        torch.backends.cudnn.benchmark = True
    elif len(gpu_list) > 1:
        model = nn.DataParallel(model, device_ids=gpu_list)
        model = convert_model(model).cuda()
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
    logger.info('device: {}, gpu_list: {}'.format(device, gpu_list))

    optimizer = create_optimizer(cfg, model)

    # optionally initialize from a checkpoint
    if args.initial_checkpoint and os.path.isfile(args.initial_checkpoint):
        load_checkpoint(model, args.initial_checkpoint)

    # optionally resume from a checkpoint
    resume_state = None
    resume_epoch = None
    if args.resume and os.path.isfile(args.resume):
        resume_state, resume_epoch = resume_checkpoint(model, args.resume)
    if resume_state and not args.no_resume_opt:
        if 'optimizer' in resume_state:
            optimizer.load_state_dict(resume_state['optimizer'])
            logger.info('Restoring optimizer state from [{}]'.format(args.resume))

    start_epoch = 0
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch

    model_ema = None
    if cfg.SOLVER.EMA:
        # Important to create EMA model after cuda()
        model_ema = ModelEma(model, decay=cfg.SOLVER.EMA_DECAY, device=device, resume=args.resume)

    lr_scheduler, num_epochs = create_scheduler(cfg, optimizer)
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    # summary
    print('='*60)
    print(cfg)
    print('='*60)
    print(model)
    print('='*60)
    summary(model, (3, cfg.INPUT.IMG_SIZE, cfg.INPUT.IMG_SIZE))

    # dataset
    dataset_train = Dataset(cfg.DATASETS.TRAIN)
    dataset_valid = Dataset(cfg.DATASETS.TEST)
    train_loader = create_loader(dataset_train, cfg, is_training=True)
    valid_loader = create_loader(dataset_valid, cfg, is_training=False)

    # loss function
    if cfg.SOLVER.LABEL_SMOOTHING > 0:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=cfg.SOLVER.LABEL_SMOOTHING).to(device)
        validate_loss_fn = nn.CrossEntropyLoss().to(device)
    else:
        train_loss_fn = nn.CrossEntropyLoss().to(device)
        validate_loss_fn = train_loss_fn

    eval_metric = cfg.SOLVER.EVAL_METRIC
    best_metric = None
    best_epoch = None
    saver = CheckpointSaver(checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=True if eval_metric == 'loss' else False)
    try:
        for epoch in range(start_epoch, num_epochs):
            train_metrics = train_epoch(
                epoch, model, train_loader, optimizer, train_loss_fn, cfg, logger,
                lr_scheduler=lr_scheduler, saver=saver, device=device, model_ema=model_ema)

            eval_metrics = validate(epoch, model, valid_loader, validate_loss_fn, cfg, logger)

            if model_ema is not None:
                ema_eval_metrics = validate(
                    epoch, model_ema.ema, valid_loader, validate_loss_fn, cfg, logger)
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            update_summary(
                epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                write_header=best_metric is None)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(
                    model, optimizer, cfg,
                    epoch=epoch, model_ema=model_ema, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_epoch(epoch, model, loader, optimizer, loss_fn, cfg, logger,
                lr_scheduler=None, saver=None, device='cuda', model_ema=None):

    losses_m = AverageMeter()
    acc_m = AverageMeter()

    model.train()

    total_time = 0.0
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time = time.time() - end
        input, target = input.to(device), target.to(device)

        output = model(input)

        loss = loss_fn(output, target)
        prec1 = accuracy(output, target, topk=(1,))[0]
        losses_m.update(loss.item(), input.size(0))
        acc_m.update(prec1.item(), output.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        batch_time = time.time() - end
        total_time += batch_time
        if last_batch or batch_idx % cfg.SOLVER.LOG_PERIOD == 0:
            lrl = [param['lr'] for param in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            output_log = 'Train: {} [{:>4d}/{} ({:>3.0f}%)] | Batch: {:.3f}s | Total: {:.0f}min | ' \
                         'Loss: {:.3f} | Acc: {:.2%} | LR: {:.3e} | Data: {:.3f}s' \
                .format(epoch, batch_idx+1, len(loader), 100. * (batch_idx+1) / len(loader),
                        batch_time, total_time / 60., losses_m.avg, acc_m.avg, lr, data_time)
            logger.info(output_log)
            sys.stdout.flush()

        if saver is not None and cfg.SOLVER.CHECKPOINT_PERIOD and (
                last_batch or (batch_idx + 1) % cfg.SOLVER.CHECKPOINT_PERIOD == 0):
            saver.save_recovery(
                model, optimizer, cfg, epoch, model_ema=model_ema, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg), ('acc', acc_m.avg)])


def validate(epoch, model, loader, loss_fn, cfg, logger, device='cuda'):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input, target = input.to(device), target.to(device)

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            torch.cuda.synchronize()

            losses_m.update(loss.item(), input.size(0))
            prec1_m.update(prec1.item(), output.size(0))
            prec5_m.update(prec5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if last_batch:
                output_log  = 'Eval: {} [{}] | Batch: {:.3f}s | Loss: {:.3f} | Prec@1: {:.2%} | Prec@5: {:.2%}' \
                    .format(epoch, len(loader), batch_time_m.avg, losses_m.avg, prec1_m.avg, prec5_m.avg)
                logger.info(output_log)
                sys.stdout.flush()

    metrics = OrderedDict([('loss', losses_m.avg), ('prec1', prec1_m.avg), ('prec5', prec5_m.avg)])
    return metrics


if __name__ == '__main__':
    main()

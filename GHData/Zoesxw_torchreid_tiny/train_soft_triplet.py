import sys
import os
import os.path as osp
import time
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from default_config import get_default_config, imagedata_kwargs, optimizer_kwargs, lr_scheduler_kwargs
from torchreid.data import ImageDataManager
from torchreid.losses import CrossEntropyLoss, TripletLoss
from torchreid.metrics import accuracy, compute_distance_matrix, evaluate_rank
from torchreid.models import build_model
from torchreid.optim import build_lr_scheduler, build_optimizer

from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.loggers import Logger
from torchreid.utils.reidtools import visualize_ranked_results
from torchreid.utils.rerank import re_ranking
from torchreid.utils.tools import check_isfile, set_random_seed
from torchreid.utils.torchtools import save_checkpoint, load_pretrained_weights, open_all_layers,\
    open_specified_layers, resume_from_checkpoint


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', type=str, default='', help='path to config file')
    parser.add_argument('--gpu-devices', type=str, default='', )
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Modify config options using the command-line')
    args = parser.parse_args()

    cfg = get_default_config()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))
    print('Show configuration\n{}\n'.format(cfg))
    torch.backends.cudnn.benchmark = True

    datamanager = ImageDataManager(**imagedata_kwargs(cfg))
    trainloader, queryloader, galleryloader = datamanager.return_dataloaders()
    print('Building model: {}'.format(cfg.model.name))
    model = build_model(cfg.model.name, datamanager.num_train_pids, 'triplet', pretrained=cfg.model.pretrained)

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    model = nn.DataParallel(model).cuda()

    criterion_t = TripletLoss(margin=cfg.loss.triplet.margin)
    criterion_x = CrossEntropyLoss(datamanager.num_train_pids, label_smooth=cfg.loss.softmax.label_smooth)
    optimizer = build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = build_lr_scheduler(optimizer, **lr_scheduler_kwargs(cfg))

    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(cfg.model.resume, model, optimizer=optimizer)

    if cfg.test.evaluate:
        distmat = evaluate(model, queryloader, galleryloader, dist_metric=cfg.test.dist_metric,
                           normalize_feature=cfg.test.normalize_feature, rerank=cfg.test.rerank, return_distmat=True)
        if cfg.test.visrank:
            visualize_ranked_results(distmat, datamanager.return_testdataset(), 'image', width=cfg.data.width,
                                     height=cfg.data.height, save_dir=osp.join(cfg.data.save_dir, 'visrank'))
        return

    time_start = time.time()
    print('=> Start training')
    for epoch in range(cfg.train.start_epoch, cfg.train.max_epoch):
        train(epoch, cfg.train.max_epoch, model, criterion_t, criterion_x, optimizer, trainloader,
              fixbase_epoch=cfg.train.fixbase_epoch, open_layers=cfg.train.open_layers)
        scheduler.step()
        if (epoch + 1) % cfg.test.eval_freq == 0 or (epoch + 1) == cfg.train.max_epoch:
            rank1 = evaluate(model, queryloader, galleryloader, dist_metric=cfg.test.dist_metric,
                             normalize_feature=cfg.test.normalize_feature, rerank=cfg.test.rerank)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'rank1': rank1,
                'optimizer': optimizer.state_dict(),
            }, cfg.data.save_dir)
    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))


def train(epoch, max_epoch, model, criterion_t, criterion_x, optimizer, trainloader, fixbase_epoch=0, open_layers=None):
    losses_t = AverageMeter()
    losses_x = AverageMeter()
    accs = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.train()
    if (epoch + 1) <= fixbase_epoch and open_layers is not None:
        print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch + 1, fixbase_epoch))
        open_specified_layers(model, open_layers)
    else:
        open_all_layers(model)
    num_batches = len(trainloader)
    end = time.time()
    for batch_idx, (imgs, pids, _, _) in enumerate(trainloader):
        data_time.update(time.time() - end)
        imgs = imgs.cuda()
        pids = pids.cuda()
        optimizer.zero_grad()
        outputs, features = model(imgs)
        loss_t = criterion_t(features, pids)
        loss_x = criterion_x(outputs, pids)
        loss = loss_t + loss_x
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        losses_t.update(loss_t.item(), pids.size(0))
        losses_x.update(loss_x.item(), pids.size(0))
        accs.update(accuracy(outputs, pids)[0].item())
        if (batch_idx + 1) % 20 == 0:
            eta_seconds = batch_time.avg * (num_batches - (batch_idx + 1) + (max_epoch - (epoch + 1)) * num_batches)
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_t {loss_t.val:.4f} ({loss_t.avg:.4f})\t'
                  'Loss_x {loss_x.val:.4f} ({loss_x.avg:.4f})\t'
                  'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                  'Lr {lr:.6f}\t'
                  'eta {eta}'.format(
                epoch + 1, max_epoch, batch_idx + 1, num_batches,
                batch_time=batch_time,
                data_time=data_time,
                loss_t=losses_t,
                loss_x=losses_x,
                acc=accs,
                lr=optimizer.param_groups[0]['lr'],
                eta=eta_str))
        end = time.time()


def evaluate(model, queryloader, galleryloader, dist_metric='euclidean', normalize_feature=False, rerank=False,
             return_distmat=False):
    batch_time = AverageMeter()
    model.eval()
    with torch.no_grad():
        print('Extracting features from query set ...')
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(queryloader):
            imgs = imgs.cuda()
            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        print('Extracting features from gallery set ...')
        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(galleryloader):
            imgs = imgs.cuda()
            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)
            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

    if normalize_feature:
        print('Normalzing features with L2 norm ...')
        qf = F.normalize(qf, p=2, dim=1)
        gf = F.normalize(gf, p=2, dim=1)

    print('Computing distance matrix with metric={} ...'.format(dist_metric))
    distmat = compute_distance_matrix(qf, gf, dist_metric)
    distmat = distmat.numpy()

    if rerank:
        print('Applying person re-ranking ...')
        distmat_qq = compute_distance_matrix(qf, qf, dist_metric)
        distmat_gg = compute_distance_matrix(gf, gf, dist_metric)
        distmat = re_ranking(distmat, distmat_qq, distmat_gg)

    print('Computing CMC and mAP ...')
    cmc, mAP = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids)
    print('** Results **')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in [1, 5, 10, 20]:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))

    if return_distmat:
        return distmat

    return cmc[0]


if __name__ == '__main__':
    main()

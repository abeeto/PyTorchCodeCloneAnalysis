import os
import yaml
import copy
import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

import utils
import models
import datasets

# loss function
def capsule_loss(with_reconstruction=False, rc_loss_weight=0.0005):
    def loss_constructor(pred, target, rc=None, rc_target_img=None):
        digit_margin_loss = models.DigitExistenceMarginLoss()
        if with_reconstruction:
            mse_loss = torch.nn.MSELoss(reduction='sum')
            return digit_margin_loss(pred, target) + rc_loss_weight*mse_loss(rc, rc_target_img) / rc.size(0)
        else:
            return digit_margin_loss(pred, target)
    return loss_constructor


# accuracy function
def single_digit_accuracy(pred, target):
    return torch.eq(pred.argmax(dim=1), target.argmax(dim=1)).sum().float() / pred.size(0)

# Train Epoch
def train(model, train_loader, optim, criterion, logger, epoch, use_cuda):
    accum_acc = 0
    accum_loss = 0
    num_batch = len(train_loader)
    model.train()
    for num_iter, (img, target) in enumerate(train_loader):
        if use_cuda:
            img, target = img.cuda(non_blocking=True), target.cuda(non_blocking=True)
        caps, pred, rc = model(img, rc_target=target)
        loss = criterion(pred, target, rc=rc, rc_target_img=img)
        optim.zero_grad()
        loss.backward()
        optim.step()
        acc = single_digit_accuracy(pred.detach(), target.detach())
        accum_acc += acc.item()
        accum_loss += loss.item()
        logger.add_scalar('train_acc_iter', acc.item(), (epoch-1)*num_batch+num_iter)
        logger.add_scalar('train_loss_iter', loss.item(), (epoch-1)*num_batch+num_iter)
        if num_iter % (num_batch // 20) == 0:
            print(f"Epoch {epoch} {num_iter * 100 // num_batch} % train accuracy {acc.item():.4f}")
            print(f"Epoch {epoch} {num_iter * 100 // num_batch} % train loss {loss.item():.4f}")

    return accum_acc / num_batch, accum_loss / num_batch

# Test Epoch
def test(model, test_loader, criterion, logger, epoch, use_cuda):
    model.eval()
    accum_acc = 0
    accum_loss = 0
    num_batch = len(test_loader)
    with torch.no_grad():
        for num_iter, (img, target) in enumerate(test_loader):
            if use_cuda:
                img, target = img.cuda(non_blocking=True), target.cuda(non_blocking=True)
            caps, pred, rc = model(img, rc_target=target)
            loss = criterion(pred, target, rc=rc, rc_target_img=img)
            acc = single_digit_accuracy(pred, target)
            accum_acc += acc.item()
            accum_loss += loss.item()
            logger.add_scalar('test_acc_iter', acc.item(), (epoch-1)*num_batch+num_iter)
            logger.add_scalar('test_loss_iter', loss.item(), (epoch-1)*num_batch+num_iter)
            if num_iter == 0:
                logger.add_images('recon/network', rc.clamp(0, 1), epoch - 1, dataformats='NCHW')
                logger.add_images('recon/target', img, epoch - 1, dataformats='NCHW')
    return accum_acc / num_batch, accum_loss / num_batch

def run(cfg):
    num_epochs = cfg['num_epochs']
    batch_size = cfg['batch_size']
    epoch = 0

    # Dataset and loader
    mnist_train, mnist_test = datasets.get_shift_MNIST()
    train_loader = data.DataLoader(mnist_train, batch_size=batch_size,
                                   **cfg['dataloader'])
    test_loader = data.DataLoader(mnist_test, batch_size=16,
                                  **cfg['dataloader'])

    # Models
    model = models.CapsNet(route_iters=cfg['route_iters'],
                           with_reconstruction=cfg['loss']['with_reconstruction'])
    best_model = copy.deepcopy(model)

    # CUDA
    if cfg['use_cuda'] and torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = model.cuda()
    else:
        cfg['use_cuda'] = False

    # Optimizer and scheduler
    optim = torch.optim.Adam(model.parameters(), lr=cfg['optim']['lr'])
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, cfg['optim']['exp_decay'], last_epoch=epoch)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, cfg['optim']['milestone'], cfg['optim']['step_decay'], last_epoch=-1)

    # Best metrics
    best_metrics = {'acc': 0.0, 'loss': float('inf')}

    # Load State Dict
    if cfg['checkpoint']['use_checkpoint'] and os.path.exists(os.path.join(cfg['checkpoint']['log_dir'], cfg['checkpoint']['model_path'])):
        epoch += 1
        print('Load Checkpoint')
        epoch, model, optim, best_model, best_metrics = utils.load_state(os.path.join(cfg['checkpoint']['log_dir'], cfg['checkpoint']['model_path']), model, optim, best_model, use_best=cfg['checkpoint']['use_best'])

    # init logger
    logger = SummaryWriter(cfg['checkpoint']['log_dir'], purge_step=epoch)

    # Loss
    criterion = capsule_loss(cfg['loss']['with_reconstruction'],
                             cfg['loss']['rc_loss_weight'])

    model = copy.deepcopy(best_model)

    for epoch in range(epoch, num_epochs):
        train_acc, train_loss = train(model, train_loader, optim, criterion, logger, epoch, cfg['use_cuda'])

        logger.add_scalar('lr', optim.param_groups[0]['lr'], epoch)
        logger.add_scalar('epoch_acc/train', train_acc, epoch)
        logger.add_scalar('epoch_loss/train', train_loss, epoch)
        print(f"Epoch {epoch} train accuracy {train_acc:.4f}")
        print(f"Epoch {epoch} train loss {train_loss:.4f}")
        test_acc, test_loss = test(model, test_loader, criterion, logger, epoch, cfg['use_cuda'])
        print(f"Epoch {epoch} test accuracy {test_acc:.4f}")
        print(f"Epoch {epoch} test loss {test_loss:.4f}")
        logger.add_scalar('epoch_acc/test', test_acc, epoch)
        logger.add_scalar('epoch_loss/test', test_loss, epoch)
        scheduler.step()
        # Save best metrics
        if best_metrics['acc'] < test_acc:
            best_model = model
            best_metrics['acc'] = test_acc
            best_metrics['loss'] = test_loss
        utils.save_state(os.path.join(cfg['checkpoint']['log_dir'],
                                      cfg['checkpoint']['model_path']),
                         model, optim, epoch, best_model, best_metrics)
    logger.flush()


if __name__ == "__main__":
    import argparser
    parser = argparser.ArgumentParser()
    parser.add_argument('config', required=True, type=str, help="Config Yaml file")
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, yaml.Loader)
    run(cfg)

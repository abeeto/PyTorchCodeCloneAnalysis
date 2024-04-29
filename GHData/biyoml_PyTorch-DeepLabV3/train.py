import os
import argparse
import torch
import utils.schedulers
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from utils.data.dataloader import create_dataloader
from utils.constants import VOID_LABEL
from utils.misc import load_config, build_model
from utils.metrics import Mean, ConfusionMatrix


class CheckpointManager(object):
    def __init__(self, logdir, model, optim, scaler, scheduler, best_score):
        self.epoch = 0
        self.logdir = logdir
        self.model = model
        self.optim = optim
        self.scaler = scaler
        self.scheduler = scheduler
        self.best_score = best_score

    def save(self, filename):
        data = {
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_score': self.best_score,
        }
        torch.save(data, os.path.join(self.logdir, filename))

    def restore(self, filename):
        data = torch.load(os.path.join(self.logdir, filename))
        self.model.load_state_dict(data['model_state_dict'])
        self.optim.load_state_dict(data['optim_state_dict'])
        self.scaler.load_state_dict(data['scaler_state_dict'])
        self.scheduler.load_state_dict(data['scheduler_state_dict'])
        self.epoch = data['epoch']
        self.best_score = data['best_score']

    def restore_lastest_checkpoint(self):
        if os.path.exists(os.path.join(self.logdir, 'last.pth')):
            self.restore('last.pth')
            print("Restore the last checkpoint.")


def get_lr(optim):
    for param_group in optim.param_groups:
        return param_group['lr']


def train_step(images, annos, model, loss_fn, optim, amp, scaler, metrics, device):
    images = images.to(device)
    annos = annos.to(device)

    optim.zero_grad()
    with autocast(enabled=amp):
        logits = model(images)
        loss = loss_fn(logits, annos)
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()

    loss = loss.item()
    metrics['loss'].update(loss, images.shape[0])


def test_step(images, annos, model, loss_fn, amp, metrics, device):
    images = images.to(device)
    annos = annos.to(device)

    with autocast(enabled=amp):
        logits = model(images)
        loss = loss_fn(logits, annos)
    preds = torch.argmax(logits, axis=1)
    loss = loss.item()

    metrics['loss'].update(loss, images.shape[0])
    metrics['cm'].update(preds, annos)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', type=str, required=True,
                        help="config file")
    parser.add_argument('--logdir', type=str, required=True,
                        help="log directory")
    parser.add_argument('--workers', type=int, default=4,
                        help="number of dataloader workers")
    parser.add_argument('--resume', action='store_true',
                        help="resume training")
    parser.add_argument('--no_amp', action='store_true',
                        help="disable automatic mix precision")
    parser.add_argument('--val_period', type=int, default=1,
                        help="number of epochs between successive validation")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = load_config(args.cfg)
    enable_amp = (not args.no_amp)

    if os.path.exists(args.logdir) and (not args.resume):
        raise ValueError("Log directory %s already exists. Specify --resume "
                         "in command line if you want to resume the training."
                         % args.logdir)

    model = build_model(cfg)
    model.to(device)

    train_loader = create_dataloader(cfg.train_csv,
                                     batch_size=cfg.batch_size,
                                     image_size=cfg.input_size,
                                     augment=True,
                                     shuffle=True,
                                     num_workers=args.workers)
    val_loader = create_dataloader(cfg.val_csv,
                                   batch_size=cfg.batch_size,
                                   image_size=cfg.input_size,
                                   num_workers=args.workers)

    loss_fn = CrossEntropyLoss(ignore_index=VOID_LABEL)
    optim = getattr(torch.optim, cfg.optim.pop('name'))(model.parameters(), **cfg.optim)
    if hasattr(torch.optim.lr_scheduler, cfg.scheduler.name):
        scheduler_class = getattr(torch.optim.lr_scheduler, cfg.scheduler.pop('name'))
    else:
        scheduler_class = getattr(utils.schedulers, cfg.scheduler.pop('name'))
    scheduler = scheduler_class(optim, **cfg.scheduler)
    scaler = GradScaler(enabled=enable_amp)

    metrics = {'loss': Mean(), 'cm': ConfusionMatrix(cfg.num_classes)}

    # Checkpointing
    ckpt = CheckpointManager(args.logdir,
                             model=model,
                             optim=optim,
                             scaler=scaler,
                             scheduler=scheduler,
                             best_score=0.)
    ckpt.restore_lastest_checkpoint()

    # TensorBoard writers
    writers = {
        'train': SummaryWriter(os.path.join(args.logdir, 'train')),
        'val': SummaryWriter(os.path.join(args.logdir, 'val'))
    }

    # Kick off
    for epoch in range(ckpt.epoch + 1, cfg.epochs + 1):
        print("-" * 10)
        print("Epoch: %d/%d" % (epoch, cfg.epochs))

        lr = get_lr(optim)
        writers['train'].add_scalar('Learning rate', lr, epoch)
        print("Learning rate:", lr)

        # Train
        model.train()
        metrics['loss'].reset()
        pbar = tqdm(train_loader,
                    bar_format="{l_bar}{bar:20}{r_bar}",
                    desc="Training")
        for (images, annos) in pbar:
            train_step(images,
                       annos,
                       model=model,
                       loss_fn=loss_fn,
                       optim=optim,
                       amp=enable_amp,
                       scaler=scaler,
                       metrics=metrics,
                       device=device)
            scheduler.step()
            pbar.set_postfix(loss='%.5f' % metrics['loss'].result)
        writers['train'].add_scalar('Loss', metrics['loss'].result, epoch)

        # Validation
        if epoch % args.val_period == 0:
            model.eval()
            metrics['loss'].reset()
            metrics['cm'].reset()
            pbar = tqdm(val_loader,
                        bar_format="{l_bar}{bar:20}{r_bar}",
                        desc="Validation")
            with torch.no_grad():
                for (images, annos) in pbar:
                    test_step(images,
                              annos,
                              model=model,
                              loss_fn=loss_fn,
                              amp=enable_amp,
                              metrics=metrics,
                              device=device)
                    pbar.set_postfix(loss='%.5f' % metrics['loss'].result)
            mIoU = metrics['cm'].IoUs.mean()
            if mIoU > ckpt.best_score:
                ckpt.best_score = mIoU
                ckpt.save('best.pth')
            print("mIoU: %.3f (best: %.3f)" % (mIoU, ckpt.best_score))
            writers['val'].add_scalar('Loss', metrics['loss'].result, epoch)
            writers['val'].add_scalar('mIoU', mIoU, epoch)

        ckpt.epoch += 1
        ckpt.save('last.pth')

    writers['train'].close()
    writers['val'].close()


if __name__ == '__main__':
    main()

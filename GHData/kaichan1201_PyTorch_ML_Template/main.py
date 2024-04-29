import os
import argparse
import torch
import torch.nn as nn
import torch.optim

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dataset import CustomData
from model import BaseModel
from train import train, validate, test

def get_arguments():
    parser = argparse.ArgumentParser()
    # mode
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    # data
    parser.add_argument('--data_dir', type=str, default='./data')
    # train options
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--epoch', '-e', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_epoch', type=int, default=10)
    # optimizer
    parser.add_argument('--optim',type=str, choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--scheduler',type=str, choices=['poly', 'linear', 'step'], default='poly')
    parser.add_argument('--warm_up_epoch', type=int, default=10)
    # load
    parser.add_argument('--load_ckpt', type=str, default='')
    # save
    parser.add_argument('--save_dir', type=str, default='./ckpt')
    parser.add_argument('--test_out_csv_path', type=str, default='pred.csv')
    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments
    args = get_arguments()

    # prepare GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare writer
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.save_dir)
    
    # prepare model
    network = BaseModel().to(device)

    # load ckpt
    best_acc = 0
    if len(args.load_ckpt) > 0:
        ckpt = torch.load(args.load_ckpt)
        network.load_state_dict(ckpt['model'])
        best_acc = ckpt['best_acc']
        print('Loaded ckpt {}, best Acc: {}'.format(args.load_ckpt, best_acc))

    if args.train:
        # prepare dataloader
        train_loader = DataLoader(dataset=CustomData('train', dir_path=args.data_dir),
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True)

        val_loader = DataLoader(dataset=CustomData('val', dir_path=args.data_dir),
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=False)
        # prepare optimizer
        optimizer = None
        if args.optim == 'sgd':
            optimizer = torch.optim.SGD([
                {'params': network.parameters()}
            ], lr=args.lr, weight_decay=1e-4, momentum=0.9)
        elif args.optim == 'adam':
            optimizer = torch.optim.Adam([
                {'params': network.parameters()}
            ], lr=args.lr, weight_decay=1e-5)

        scheduler = None
        if args.scheduler == 'poly':
            func = lambda epoch: (1 - epoch / args.n_epochs)**0.9  # poly
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, func)

        # prepare criterion(s)
        criterion_cls = nn.CrossEntropyLoss()

        # start training
        for e in range(1, args.epoch + 1):
            # adjust lr
            if e <= args.warm_up_epoch:
                for g in optimizer.param_groups:
                    g['lr'] = args.lr * e / args.warn_up_epoch

            # train
            train(model=network, train_loader=train_loader, criterion_cls=criterion_cls, optimizer=optimizer,
                  device=device, writer=writer, cur_epoch=e)

            if e % args.val_epoch == 0:
                # validation
                val_acc = validate(model=network, val_loader=val_loader, device=device, writer=writer,
                                   cur_epoch=e)
                if val_acc > best_acc:
                    best_acc = val_acc
                    # save ckpt
                    torch.save({
                        'model': network.state_dict(),
                        'best_acc': best_acc
                    }, os.path.join(args.save_dir, 'best.pth'))

            if e > args.warm_up_epoch:
                scheduler.step()

    if args.test:
        # prepare dataloader
        test_loader = DataLoader(dataset=CustomData('test', dir_path=args.data_dir),
                                 batch_size=1,
                                 num_workers=args.num_workers,
                                 shuffle=False)

        test(model=network, test_loader=test_loader, device=device, out_path=args.test_out_csv_path)

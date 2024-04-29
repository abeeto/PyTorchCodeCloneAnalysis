import torch
import argparse
import os
from utils import setup_seed, get_logger, AverageMeter, get_lr, save_model, get_remain_time
from model import Model
from torch.utils.data import DataLoader
from dataset import TrainSet, ValidationSet
from tensorboardX import SummaryWriter
from scipy.stats import spearmanr
import time


def train(args, model, optimizer, train_loader, device, criterion):
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    model.train()
    max_iter = len(train_loader)

    for idx, (image, label) in enumerate(train_loader):
        start = time.time()
        n = image.size(0)

        image = image.to(device)
        label = label.to(device)
        output = model(image)
        optimizer.zero_grad()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - start)
        remain_time = get_remain_time(idx, max_iter, batch_time.avg)
        print('\r{}: {}/{}, loss: {:.4f} [remain: {}]'.format(train.__name__, idx+1, len(train_loader), loss_meter.avg, remain_time), end='', flush=True)
    print()
    return loss_meter.avg


def validation(model, test_loader, device, criterion):
    model.eval()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    max_iter = len(test_loader)
    predict = []
    labels = []

    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            start = time.time()
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            loss = criterion(output, label)
            loss_meter.update(loss.item())

            predict.append(output.view(-1).detach().item())
            labels.append(label.view(-1).detach().item())

            batch_time.update(time.time() - start)
            remain_time = get_remain_time(i, len(test_loader), batch_time.avg)
            print('\r{}: {}/{} [remain: {}]'.format(train.__name__, i + 1, len(test_loader), remain_time), end='',
                  flush=True)

    corr, _ = spearmanr(predict, labels)
    return loss_meter.avg, corr


def main():
    args = parser.parse_args()
    logger = get_logger(os.path.join(args.log_path, args.name, 'log.txt'))
    logger.info(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    setup_seed(args.seed)
    torch.set_num_threads(1)
    os.makedirs(os.path.join(args.log_dir, args.name, 'tensorboard'), exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, args.name, 'checkpoint'), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('==> Preparing data..')
    train_set = TrainSet(args.train_data)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    val_set = ValidationSet(args.val_data)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=args.num_workers)

    print('==> Building model..')
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    start_epoch = 0

    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), f'No checkpoint found at {args.resume}'
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

    model = model.to(device)
    writer = SummaryWriter(args.tensorboard)
    criterion = torch.nn.MSELoss()
    epoch_meter = AverageMeter()

    for epoch in range(start_epoch, start_epoch + args.epoch):
        start = time.time()
        train_loss = train(args, model, optimizer, train_loader, device, criterion)
        writer.add_scalar('Train/Loss', train_loss, global_step=epoch)
        writer.add_scalar('Train/LearningRate', scalar_value=get_lr(optimizer), global_step=epoch)
        logger.info('train - epoch: {}, loss: {}'.format(epoch, train_loss))
        if epoch % 5 == 0:
            val_loss, val_srocc = validation(model, val_loader, device, criterion)
            writer.add_scalar('Test/Loss', val_loss, global_step=epoch)
            writer.add_scalar('Test/SROCC', val_srocc, global_step=epoch)
            logger.info('validation - epoch: {}, loss: {:.4f}, SROCC: {:.4f}'.format(epoch, val_loss, val_srocc))
        writer.flush()
        if epoch % 500 == 0 and epoch > 0:
            save_model(model, optimizer, epoch, args)
        epoch_meter.update(time.time() - start)
        remain_time = get_remain_time(epoch, args.epoch, epoch_meter.avg)
        print('epoch: {}/{} [remain: {}]'.format(epoch + 1, args.epoch, remain_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data', type=str)
    parser.add_argument('--val_data', type=str)
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--name', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--lr', type=float)

    parser.add_argument('--log_dir', type=str, default='experiments', help='The log path saved')
    parser.add_argument('--resume', type=str, default=None,
                        help='The checkpoint path used to continue the train')
    parser.add_argument('--num_workers', type=int, default=4, help='The num of thread used to load data')
    main()
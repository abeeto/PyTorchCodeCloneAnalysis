import os
import argparse
import time
import model
from warmup_scheduler import GradualWarmupScheduler
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import optim
from tqdm import tqdm

from model.transformer_module import TransformerEncoder, TransformerBlock
from data import get_dataloaders
from data.transformation import train_transform, val_transform, auto_transform
from utils import AverageMeter, EarlyStopping, Lookahead, ProgressMeter, accuracy, get_lr

parser = argparse.ArgumentParser(description = 'PyTorch Medical Project')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None,
                    help="Select specific GPU to run the model")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='Input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='Input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='Number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='Learning rate (default: 0.01)')
parser.add_argument('--warmup-epoch', type=int, default=10,
                    help='Warmup epoch (default: 10)')
parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                    help='Learning rate step gamma for StepLR (default: 0.1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training(default: False)')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='Quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='Random seed (default: 1)')
parser.add_argument('--early-stop', type=int, default=10,
                    help="After n consecutive epochs,val_loss isn't improved then early stop")
parser.add_argument('--lookahead', action = "store_true",
                    help='Use lookahead optimizer wraper (default: False)')
parser.add_argument('--model-path', type=str, default="checkpoint.pt",
                    help='For Saving the current Model(default: checkpoint.pt)')
parser.add_argument('--wandb-name', type=str, default="",
                    help='Setting run name for wandb (default: "").')
parser.add_argument('--train-data-path', type=str, default="./data/retina/train",
                    help='Path for the training data (default: ./data/retina/train')
parser.add_argument('--test-data-path', type=str, default="./data/retina/train",
                    help='Path for the testing data (default: ./data/retina/train')


WEIGHTS_PATH = "./weights"
if not os.path.exists(WEIGHTS_PATH):
    os.makedirs(WEIGHTS_PATH)


def adjust_lr(args, optimizer, epoch):
    lr = args.lr * 0.1**((epoch + 1) // 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_MyDataset(args, model, device, optimizer, train_loader, val_loader):
    early_stop = EarlyStopping(
        patience = args.early_stop,
        verbose = True,
        delta = 1e-3,
        path = os.path.join(WEIGHTS_PATH, args.model_path)
    )

    scheduler_steplr = optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = args.gamma)
    scheduler_warmup = GradualWarmupScheduler(
        optimizer,
        multiplier = 10,
        total_epoch = args.warmup_epoch,
        after_scheduler = scheduler_steplr
    )

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, args.epochs + 1):
        # Train model
        train_losses = AverageMeter('Train Loss', ':.4e')
        train_top1 = AverageMeter('Acc@1', ':6.2f')
        model.train()
        scheduler_warmup.step(epoch)

        for _, data_dict in tqdm(enumerate(train_loader)):
            optimizer.zero_grad(set_to_none = True)
            data, target = data_dict['image'].to(device), data_dict['targets'].to(device)
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = F.cross_entropy(output, target)
            train_losses.update(loss.item(), data.size(0))
            # loss.backward()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc1 = accuracy(output, target)
            train_top1.update(acc1[0].item(), output.size(0))
        wandb.log({'Lr/train' : optimizer.param_groups[0]['lr']})

        # Validate model
        batch_time = AverageMeter('Time', ':6.3f')
        val_loss = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        progress = ProgressMeter(
            num_batches = len(val_loader),
            meters = [top1, val_loss, batch_time],
            prefix = "Epoch: [{}]".format(epoch),
            batch_info = "  Iter: "
        )

        model.eval()
        end = time.time()
        for it, data_dict in enumerate(val_loader):
            data, target = data_dict['image'].to(device), data_dict['targets'].to(device)

            # Compute output and loss
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss

            # Measure accuracy,  update top1@ and losses
            acc1 = accuracy(output, target)
            val_loss.update(loss, data.size(0))
            top1.update(acc1[0].item(), data.size(0))
            progress.display(it)

            # Measure the elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        wandb.log({
            "Loss/train" : train_losses.avg,
            "Loss/val" : val_loss.avg,
            "Acc/train" : train_top1.avg,
            "Acc/val" : top1.avg
        })

        early_stop(val_loss.avg, model)
        if early_stop.early_stop_flag:
            print(f"Epoch [{epoch} / {args.epochs}]: early stop")
            break

    ckpt = torch.load(early_stop.path)
    model.load_state_dict(ckpt)
    return model, train_losses.avg, early_stop.best_score


def test_MyDataset(model, device, test_loader):
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()
    with torch.no_grad():
        for _, data_dict in enumerate((test_loader)):
            data, target = data_dict['image'].to(device), data_dict['targets'].to(device)

            # Compute output and loss
            output = model(data)

            # Measure accuracy,  update top1@
            test_acc = accuracy(output, target)
            top1.update(test_acc[0].item(), data.size(0))
    print(f"Test accuracy: {top1.avg}")
    wandb.log({'Acc/tes' : top1.avg})


def fix_layer(m):
    for param in m.parameters():
        param.requires_grad = False


def main():
    # Training settings
    args = parser.parse_args()
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

    # Using wandb
    wandb.init(project = "Training on retina classification(Lab508)_Train")
    if args.wandb_name != "":
        wandb.run.name = args.wandb_name
    wandb.config.update(args)

    if args.use_cuda:
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if args.use_cuda else "cpu")

    train_kwargs = {'batch_size' : args.batch_size, 'shuffle' : True}
    test_kwargs = {'batch_size' : args.test_batch_size, 'shuffle' : False}
    if args.use_cuda:
        cuda_kwargs = {
            'num_workers': 4,
            'pin_memory': True
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_dl, val_dl, _ = get_dataloaders(
        train_dir = args.train_data_path,
        test_dir = args.test_data_path,
        train_transform = auto_transform,
        test_transform = val_transform,
        split = (0.8, 0.2),
        **train_kwargs
    )

    net = torchvision.models.resnet50(pretrained = True)
    net.fc = nn.Sequential(
        nn.Linear(net.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(512, 2),
        nn.Sigmoid()
    )
    net = nn.DataParallel(net)
    net.to(device)
    wandb.watch(net)

    optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum = 0.9)
    if args.lookahead:
        optimizer = Lookahead(optimizer)
    net, train_loss, val_loss = train_MyDataset(args, net, device, optimizer, train_dl, val_dl)
    print(f"Final  --->  Train loss: {train_loss}  Val loss: {val_loss}")

    test_MyDataset(net, device, val_dl)

if __name__ == '__main__':
    main()

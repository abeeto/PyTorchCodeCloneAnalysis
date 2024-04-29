import argparse
import time
import model
from warmup_scheduler import GradualWarmupScheduler
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter
from torch import optim
from tqdm import tqdm

from model.transformer_module import TransformerEncoder, TransformerBlock
from data import get_dataloaders
from data.transformation import train_transform, val_transform
from utils import AverageMeter, EarlyStopping, ProgressMeter, accuracy, get_lr

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='Input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='Input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='Number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='Learning rate (default: 0.01)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training(default: False)')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='Quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='Random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='How many batches to wait before logging training status')
parser.add_argument('--early-stop', type=int, default=10,
                    help="After n consecutive epochs,val_loss isn't improved then early stop")
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model(default: False)')

# Using wandb
wandb.init(project = "Training on retina classification(Lab508)_NEW")

# Using Tensorboard
# writer = SummaryWriter()

def adjust_lr(args, optimizer, epoch):
    lr = args.lr * 0.1**((epoch + 1) // 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#===============MyDataset========================
def train_MyDataset(args, model, device, optimizer, train_loader, val_loader):
    global writer
    early_stop = EarlyStopping(
        patience = args.early_stop,
        verbose = True,
        delta = 1e-3
    )

    for epoch in range(1, args.epochs + 1):
        # Train model
        train_losses = AverageMeter('Train Loss', ':.4e')
        train_top1 = AverageMeter('Acc@1', ':6.2f')
        model.train()
        # scheduler_steplr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
        scheduler_steplr = optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.1)
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier = 10, total_epoch = 5, after_scheduler = scheduler_steplr)
        scheduler_warmup.step(epoch)
        for _, data_dict in tqdm(enumerate(train_loader)):
            data, target = data_dict['image'].to(device), data_dict['targets'].to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            train_losses.update(loss.item(), data.size(0))
            loss.backward()
            print("*****", epoch, optimizer.param_groups[0]['lr'], '*****')
            optimizer.step()
            wandb.log({'Lr/train' : optimizer.param_groups[0]['lr']})

            acc1 = accuracy(output, target)
            train_top1.update(acc1[0].item(), output.size(0))

        adjust_lr(args, optimizer, epoch)

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
        # scheduler.step(val_loss.avg)c
        early_stop(val_loss.avg, model)
        if early_stop.early_stop_flag:
            print(f"Epoch [{epoch} / {args.epochs}]: early stop")
            break

        # Use Tensorboard to record and visualize specific important info
        dis_acc = "X-axis: epoch & Y-axis: accuracy"
        dis_loss = "X-axis: epoch & Y-axis: loss"
        dis_lr = "X-axis: epoch & Y-axis: learngin rate"
        # writer.add_scalar("Loss/train", train_losses.avg, epoch, display_name = dis_acc)
        # writer.add_scalar("Loss/val", val_loss.avg, epoch, display_name = dis_acc)
        # writer.add_scalar("Acc/train", train_top1.avg, epoch, display_name = dis_loss)
        # writer.add_scalar("Acc/val", top1.avg, epoch, display_name = dis_loss)
        # writer.add_scalar("Lr/train", get_lr(optimizer), epoch, display_name = dis_lr)

        wandb.log({
            "Loss/train" : train_losses.avg,
            "Loss/val" : val_loss.avg,
            "Acc/train" : train_top1.avg,
            "Acc/val" : top1.avg
        })

    #     for name, param in model.named_parameters():
    #         writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
    # writer.flush()

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

    wandb.config.update(args)

    if args.use_cuda:
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)

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

    path = './data/test1'
    train_dl, val_dl, test_dl = get_dataloaders(
        train_dir = path,
        test_dir = path,
        train_transform = train_transform,
        test_transform = val_transform,
        split = (0.8, 0.2),
        **train_kwargs
    )

    # net = torchvision.models.resnext50_32x4d(pretrained=True)
    # net.fc = nn.Sequential(
    #     nn.Linear(net.fc.in_features,512),
    #     nn.ReLU(),
    #     nn.Dropout(),
    #     nn.Linear(512, 2),
    #     nn.Sigmoid()
    # )
    # net = model.VIT(img_dim = 224, num_classes = 2, blocks = 12)
    net = model.VIT_B_16(num_classes = 2)
    net.to(device)
    wandb.watch(net)
    optimizer = optim.SGD(net.parameters(), lr = args.lr)

    net, train_loss, val_loss = train_MyDataset(args, net, device, optimizer, train_dl, val_dl)
    print(f"Final  --->  Train loss: {train_loss}  Val loss: {val_loss}")

    # test_MyDataset(net, device, test_dl)
    test_MyDataset(net, device, val_dl)
    # writer.close()
    if args.save_model:
        torch.save(net.state_dict(), "latest_model.pt")

if __name__ == '__main__':
    main()

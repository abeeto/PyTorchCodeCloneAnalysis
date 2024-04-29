# Import Dependencies
import os
import copy
from argparse import ArgumentParser
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
import torch.backends.cudnn as cudnn
from model import SqueezeNet
from dataset import get_data_loader
from utils import AverageMeter, accuracy


# Training Function
def train(model, train_loader, device, criterion, optimizer, scheduler):
    model.train()
    running_loss = AverageMeter()
    running_accuracy = AverageMeter()
    t = tqdm(train_loader)

    for idx, (images, labels) in enumerate(t):
        images = images.to(device)
        labels = labels.to(device)

        prediction = model(images)
        loss = criterion(prediction, labels)
        acc = accuracy(labels.data, prediction.data)

        running_loss.update(loss.item(), images.size(0))
        running_accuracy.update(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        t.set_description(f"Train Loss: {running_loss.avg:.3f}, Train Acc: {running_accuracy.avg:.3f}")

    print(f"Train Loss: {running_loss.avg:.2f}")
    print(f"Train Acc: {running_accuracy.avg:.2f}")

    return running_loss, running_accuracy


# Evaluate Function
def evaluate(model, val_loader, device, criterion):
    # Evaluate the Model
    model.eval()
    running_accuracy = AverageMeter()
    running_loss = AverageMeter()
    t = tqdm(val_loader)

    for idx, (images, labels) in enumerate(t):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            prediction = model(images)
            loss = criterion(prediction, labels)
            acc = accuracy(labels.data, prediction.data)

            running_loss.update(loss.item(), len(images))
            running_accuracy.update(acc)

            t.set_description(f"Val Loss: {running_loss.avg:.3f}, Val Acc: {running_accuracy.avg:.3f}\n")

    print(f"Val Loss: {running_loss.avg:.2f}")
    print(f"Val Acc: {running_accuracy.avg:.2f}")

    return running_loss, running_accuracy


def main(args):
    # Define device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    print("Device: ", device)

    # Set seed for reproducability
    torch.manual_seed(args.seed)

    # Setup wandb
    wandb.init(project="SqueezeNet-PyTorch")
    wandb.config.update(args)

    # Dataset Transforms
    transform = transforms.Compose(
        [transforms.Resize(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5],
                              [0.5, 0.5, 0.5]),
         ])

    # Get dataloaders
    train_loader, val_loader = get_data_loader(args.batch_size, transform, num_workers=args.num_workers)

    # Instantiate the Model
    model = SqueezeNet(in_ch=3, num_classes=10)
    model.to(device)

    # Model Loss Function
    criterion = nn.CrossEntropyLoss()

    # Model Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        steps_per_epoch=int(len(train_loader)),
        epochs=args.epochs)

    wandb.watch(model, criterion, log="all")

    # Train the Model
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, device, criterion, optimizer, scheduler)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        print(f"\nEpoch: {epoch}, Train Loss: {train_loss.avg}, Train Accuracy: {train_acc.avg}")
        print(f"\nEpoch: {epoch}, Val Loss: {val_loss.avg}, Val Accuracy: {val_acc.avg}\n")

        wandb.log({"Epoch": epoch,
                   "base_lr": args.lr,
                   "Training Loss": train_loss.avg,
                   "Training Accuracy": train_acc.avg,
                   "Validation Loss": val_acc.avg,
                   "Validation Accuracy": val_acc.avg,})

        if val_acc.avg > train_acc.avg:
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

    print('Best Epoch: {}'.format(best_epoch))
    torch.save(best_weights, os.path.join(args.dirpath_out, 'best.pth'))

    return model.load_state_dict(best_weights)


def build_parser():
    parser = ArgumentParser(prog="SqueezeNet")
    parser.add_argument("-bs", "--batch_size", default=32, required=False, type=int,
                        help="Optional. Dataset Batch Size.")
    parser.add_argument("-epochs", "--epochs", default=100, required=False, type=int,
                        help="Optional. Number of training epochs.")
    parser.add_argument("-n", "--num_workers", default=4, required=False, type=int,
                        help="Optional. Number of workers.")
    parser.add_argument("-lr", "--learning_rate", default=1e-3, required=False, type=float,
                        help="Optional. Learning Rate.")
    parser.add_argument("-seed", "--seed", default=100, required=False, type=int,
                        help="Optional. Pytorch seed for reproducability.")
    parser.add_argument("-o", "--dirpath_out", required=True, type=str,
                        help="Required. Path to directory to save best model weights.")

    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    main(args)

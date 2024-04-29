from __future__ import print_function
import torch
import torchvision

from data import data_train, data_valid
from model import Net

import argparse
from train import train
from test import test

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"


transform = torchvision.transforms.Compose([
        torchvision.transforms.transforms.ToTensor(),
        torchvision.transforms.transforms.Normalize((0.1307,), (0.3081,))
        ])


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                        help='random seed (default: 123)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    train_dataloader = torch.utils.data.DataLoader(data_train, shuffle=True, **train_kwargs)
    valid_dataloader = torch.utils.data.DataLoader(data_valid, shuffle=True, **test_kwargs)
    model = Net().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_dataloader, optimizer, epoch)
        test(model, device, valid_dataloader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "./checkpoints/mnist_cnn2.pt")


if __name__ == '__main__':
    main()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import argparse

from models import resnext
from utils import set_seed, cifar, transformations


def cmd_line_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--path', type=str, default='datasets/',
                        help='Root directory of CIFAR datasets.')
    parser.add_argument('--type', type=str, choices=['CIFAR10', 'CIFAR100'],
                        default='CIFAR10', help='Which CIFAR to use.')
    # Training
    parser.add_argument('--batch', type=int, default=128,
                        help='Mini-batch size.')
    parser.add_argument('--valid_size', type=int, default=0.9,
                        help='Size of validation set (as percentage).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum.')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--verbose_epochs', type=int, default=50,
                        help='Number of epochs after which to evaluate the model.')
    # Hardware
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of subprocesses for data loading')
    parser.add_argument('--pin_memory', type=bool, default=True,
                        help='If True copies the tensor to CUDA pinned device.')

    return parser.parse_args()


def split_indices(size, valid_size):
    indices = list(range(size))
    np.random.shuffle(indices)
    split = int(valid_size * size)
    return indices[split:], indices[:split]


def train_eval_loaders(path, batch_size, valid_size, num_workers, pin_memory, type='CIFAR10'):
    train_data = cifar(path=path, transform=transformations(True), type=type)
    eval_data = cifar(path=path, transform=transformations(False), type=type)

    train_idx, eval_idx = split_indices(len(train_data), valid_size)

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    eval_sampler = torch.utils.data.SubsetRandomSampler(eval_idx)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory)

    eval_loader = torch.utils.data.DataLoader(eval_data,
                                              batch_size=batch_size,
                                              sampler=eval_sampler,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory)

    return train_loader, eval_loader


def train(model, optimizer, dataloader, device, verbose_size=10):
    model.train()
    criterion = nn.CrossEntropyLoss()

    list_loss = []
    i = 1
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        list_loss.append(loss.item())

        loss.backward()
        optimizer.step()

        if len(list_loss) == verbose_size:
            avg_loss = sum(list_loss) / verbose_size
            print(f'[{i}/{len(dataloader)}] Average loss: {avg_loss:.4f}')
            list_loss = []
            i += 1


def eval(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    list_loss = []
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        list_loss.append(loss.item())

    avg_loss = sum(list_loss) / len(list_loss)
    print(f'[EVALUATION] Average loss: {avg_loss:.4f}\n')


def main():
    set_seed(888)
    args = cmd_line_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, eval_loader = train_eval_loaders(path=args.path,
                                                   batch_size=args.batch,
                                                   num_workers=args.num_workers,
                                                   pin_memory=args.pin_memory,
                                                   valid_size=args.valid_size,
                                                   type=args.type)

    model = resnext.resnext29()
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    n_epochs = args.n_epochs

    for epoch in range(n_epochs):
        print(f'[{epoch}/{n_epochs}] ##### NEW EPOCH #####')
        train(model, optimizer, train_loader, device)

        if epoch % args.verbose_epochs == 0:
            eval(model, eval_loader, device)


if __name__ == '__main__':
    main()

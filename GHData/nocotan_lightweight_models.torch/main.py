# -*- coding: utf-8 -*-
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from datasets import get_loaders


def train(epoch, model, optimizer, criterion, train_loader, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()

    print("Epoch: {}, Loss: {:.5f}, Acc: {:.3f}, [{:3f} sec]".format(
        epoch,
        train_loss / total,
        100. * correct / total,
        time.time() - start,
    ))

def test(model, criterion, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()

    print("Loss: {:.5f}, Acc: {:.3f}, [{:.3f} sec]".format(
        test_loss / total,
        100. * correct / total,
        time.time() - start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model", type=str, default="mobilenet_v2")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--dataroot", type=str, default="/tmp/data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default="/tmp/chkpt.pth.tar")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--pretrained", type=str, default=None)
    args = parser.parse_args()
    print(args)

    if torch.cuda.is_available():
        print("cuda is available, use cuda")
        device = torch.device("cuda")
    else:
        print("cuda is not available, use cpu")
        device = torch.device("cpu")

    print("download dataset: {}".format(args.dataset))
    train_loader, test_loader, n_classes = get_loaders(dataset=args.dataset,
                                                       root=args.dataroot,
                                                       batch_size=args.batch_size)

    print("build model: {}".format(args.model))
    if args.model == "mobilenet":
        from models import MobileNet
        model = MobileNet(n_classes=n_classes)
    elif args.model == "mobilenet_v2":
        from models import MobileNet_v2
        model = MobileNet_v2(n_classes=n_classes)
    elif args.model == "shufflenet":
        from models import ShuffleNet
        model = ShuffleNet(n_classes=n_classes)
    elif args.model == "shufflenet_v2":
        from models import ShuffleNet_v2
        model = ShuffleNet_v2(n_classes=n_classes)
    elif args.model == "squeezenet":
        from models import SqueezeNet
        model = SqueezeNet(n_classes=n_classes)
    else:
        raise NotImplementedError

    model = model.to(device)
    if args.pretrained:
        model.load_state_dict(torch.load(args.checkpoint))

    if args.n_gpus > 1:
        gpus = []
        for i in range(args.n_gpus):
            gpus.append(i)
        model = nn.DataParallel(model, device_ids=gpus)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if args.mode == "train":
        for epoch in range(args.n_epochs):
            train(epoch, model, optimizer, criterion, train_loader, device)
            if (epoch + 1) % args.save_every == 0:
                print("saving model...")
                torch.save(the_model.state_dict(), args.checkpoint)
    elif args.mode == "test":
        test(model, criterion, test_loader, device)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from net.pointnet_cls import PointnetCls
from utils.loader import ModelNet40


def parse_args():
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--dataset_path', default='./data/ModelNet40',
                        help='specify the path to dataset (default to ./data/ModelNet40)')
    parser.add_argument('--download', type=bool, default=False, help='whether to download the modelnet40')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size (default to 24')
    parser.add_argument('--trained_model', default='', help='pre-trained net path (default to none)')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training (default to 200)')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate in training (default to 0.01)')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate (default to 0.1)')
    parser.add_argument('--regularize', type=bool, default=True, help='to regularize with transform matrix')
    return parser.parse_args()


def train(model: nn.Module, loader: DataLoader, optim: optim, epoch: int, opt, device):
    model.train()
    correct = 0
    train_loss = 0
    for i, data in enumerate(loader):
        points, label = data
        # points = points.transpose(1, 2)
        points, label = points.to(device), label.to(device)
        optim.zero_grad()
        pred, trans = model(points)
        loss = F.nll_loss(pred, label)
        if opt.regularize:
            I = torch.eye(3).view(1, 3, 3).to(device)
            loss += torch.mean(torch.norm(torch.bmm(trans, trans.transpose(1, 2)) - I, dim=[1, 2])) * 0.001
        loss.backward()
        optim.step()
        pred = pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()
        train_loss += loss.item()
        print("[epoch %3d: batch %3d] train loss: %7f accuracy %7f" % (
            epoch, i, train_loss, correct / float(opt.batch_size)))


def test(model, loader, epoch, opt, device):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            points, label = data
            # points = points.transpose(1, 2)
            points, label = points.to(device), label.to(device)
            pred, trans = model(points)
            loss = F.nll_loss(pred, label)
            pred = pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
            test_loss += loss.item()
    print("[epoch %3d: batch %3d] test loss: %7f accuracy %7f" % (
        epoch, i, test_loss, correct / float(len(loader.dataset))))


def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = opt.dataset_path

    if opt.download:
        os.system("wget -O ./data/modelnet40.zip http://modelnet.cs.princeton.edu/ModelNet40.zip")
        os.system("unzip ./data/modelnet40.zip -d ./data")
        os.system("rm ./data/modelnet40.zip")
        path = "./data/ModelNet40"

    train_set = ModelNet40(path=path, npoints=1024, test=False)
    test_set = ModelNet40(path=path, npoints=1024, test=True)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=12)

    num_classes = train_set.class_num
    model = PointnetCls(num_classes).to(device)

    if opt.trained_model != "":
        model.load_state_dict(opt.trained_model)

    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=opt.decay_rate)

    for epoch in range(1, opt.epoch + 1):
        scheduler.step()
        train(model, train_loader, optimizer, epoch=epoch, opt=opt, device=device)
        test(model, test_loader, epoch=epoch, opt=opt, device=device)


if __name__ == "__main__":
    opt = parse_args()
    main(opt)

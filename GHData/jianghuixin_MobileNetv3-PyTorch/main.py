import logging
import os
from datetime import datetime

import torch
import visdom
from torch import nn, optim
from torch.utils.data import DataLoader

import models
from config import DefaultConfig
from data import dataset

opt = DefaultConfig()
logging.basicConfig(level=logging.INFO)


def __one_epoch(model: nn.Module, loader, criterion, optimizer=None):
    """
    模型迭代一步
    :param model: 使用的模型
    :param loader: 数据加载器
    :param criterion: 损失函数
    :param optimizer: 优化器, 不为 None 表示训练阶段
    :return: 这一步的损失
    """
    if optimizer:
        model.train()
    else:
        model.eval()

    steps = 1
    loss_sum = 0
    accuracy_sum = 0

    with torch.set_grad_enabled(optimizer is not None):
        for inp, label in loader:
            if optimizer is not None:
                optimizer.zero_grad()

            if opt.use_gpu:
                inp = inp.cuda()
                label = label.cuda()

            score = model(inp)
            loss = criterion(score, label)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            oup = torch.argmax(score, dim=1)
            accuracy = torch.sum(label.eq(oup)) / oup.shape[0]

            loss_sum += loss.item()
            accuracy_sum += accuracy.item()
            steps += 1

    return loss_sum / steps, accuracy_sum / steps


def vis_plot(viz, epoch, value, win):
    """
    根据 Epoch 绘制直线
    :param viz: visdom object
    :param epoch: epoch start with 0
    :param value: point value
    :param win: window name in visdom
    :return: None
    """
    viz.line(X=torch.tensor([epoch]),
             Y=torch.tensor([value]),
             win=win,
             update="append" if epoch else None,
             opts={"title": win, "xlabel": "epoch"}
             )


def train():
    # step-1: 模型
    model: nn.Module = getattr(models, opt.model)(classes=2)
    if opt.load_model_path:
        weight = torch.load(opt.load_model_path)
        model.load_state_dict(weight)

    if opt.use_gpu:
        model.cuda()

    # step-2: 数据
    data_train = dataset.CatDog(opt.data_root, train=True)
    data_valid = dataset.CatDog(opt.data_root, train=False)

    loader_train = DataLoader(data_train, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    loader_valid = DataLoader(data_valid, opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    # step-3: 目标函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    # step-4: 统计及可视化
    if opt.env:
        viz = visdom.Visdom(env=opt.env)
    loss_prev = None
    accuracy_best = 0.0

    # 训练
    for epoch in range(opt.max_epoch):
        logging.info(f"Epoch {epoch + 1}/{opt.max_epoch}")

        # 训练阶段
        loss_train, accuracy_train = __one_epoch(model, loader_train, criterion, optimizer)

        if opt.env:
            vis_plot(viz, epoch, loss_train, "Loss of train")
            vis_plot(viz, epoch, accuracy_train, "Acc of train")

        # 验证阶段
        loss_valid, accuracy_valid = __one_epoch(model, loader_valid, criterion)

        if opt.env:
            vis_plot(viz, epoch, loss_valid, "Loss of valid")
            vis_plot(viz, epoch, accuracy_valid, "Acc of valid")

        # 如果损失增大, 则降低学习率
        if loss_prev and loss_valid > loss_prev:
            lr = opt.lr * opt.lr_decay

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        # 仅保留最佳的模型
        if accuracy_valid > accuracy_best:
            weight = model.state_dict()
            weight_filename = datetime.today().strftime("%Y-%m-%d %H:%M.pth")

        loss_prev = loss_valid

    # 保存最佳模型
    torch.save(weight, os.path.join("./checkpoints", weight_filename))


if __name__ == "__main__":
    train()

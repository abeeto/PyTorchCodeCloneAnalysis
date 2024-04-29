import torch
import torch.nn as nn
from resnet18 import ResNet18
from dataset import get_dataset, get_dataloader
from utils import AverageMeter
import time


def train_one_epoch(device, model, dataloader, criterion, optimizer, epoch, total_epoch, report_freq=10):
    """
    在Deep Learning training里一个完整的epoch都要做哪些事情？
    1. 把所有batch循环一遍
    2. 每次做一个前向，一个反向，梯度下降
    """
    print(f'----- Training Epoch [{epoch}/{total_epoch}]:')
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    model.train()
    for batch_id, data in enumerate(dataloader):
        # get the inputs; data is a list of [inputs, labels]
        image, label = data[0].to(device), data[1].to(device)

        out = model(image)
        loss = criterion(out, label)

        # 先梯度清零还是最后清零，两种做法都有，官方教程有先清零的
        optimizer.zero_grad()
        loss.backward()     # 反向传播得到每个参数的梯度值（通过autograd实现）
        optimizer.step()    # 通过梯度下降执行一步参数更新
        # optimizer.zero_grad()  # 将梯度归零，即将本次的梯度记录清空

        pred = nn.functional.softmax(out, dim=1)
        # print(pred)
        # print(torch.argmax(pred, dim=1))
        # print(label)
        # print('pred.shape', pred.shape)
        # print('label.shape', label.shape)
        acc = (torch.argmax(pred, dim=1) == label).float().sum() / label.shape[0]

        batch_size = image.shape[0]
        # print(loss)

        if torch.cuda.is_available():
            loss_meter.update(loss.cpu().detach().numpy(), batch_size)
            acc_meter.update(acc.cpu().detach().numpy(), batch_size)
        else:
            loss_meter.update(loss.detach().numpy(), batch_size)
            acc_meter.update(acc.detach().numpy(), batch_size)

        if batch_id > 0 and batch_id % report_freq == 0:
            print(f'----- Batch[{batch_id}/{len(dataloader)}], Loss: {loss_meter.avg:.4}, Acc: {acc_meter.avg:.4}')

    print(f'----- Epoch[{epoch}/{total_epoch}], Training Loss: {loss_meter.avg:.4}, Acc@1: {acc_meter.avg:.4}')



def validate(device, model, dataloader, criterion, report_freq=10):
    """
    Validate和train的区别在哪？
    1. 因为不需要training，input可拿掉criterion, optimizer这俩用于训练的
    2. model.train()换成model.eval()。可加上with torch.no_grad()节约计算资源
    3. 去掉gradient descent三部曲：optimizer.zero_grad(), loss.backward(), optimizer.step()
    """
    print(f'----- Validation')
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    model.eval()
    with torch.no_grad(): # when not training, we don't need to calculate the gradients
        for batch_id, data in enumerate(dataloader):
            image, label = data[0].to(device), data[1].to(device)

            out = model(image)
            loss = criterion(out, label)

            pred = nn.functional.softmax(out, dim=1)
            acc = (torch.argmax(pred, dim=1) == label).float().sum() / label.shape[0]

            batch_size = image.shape[0]
            if torch.cuda.is_available():
                loss_meter.update(loss.cpu().detach().numpy(), batch_size)
                acc_meter.update(acc.cpu().detach().numpy(), batch_size)
            else:
                loss_meter.update(loss.detach().numpy(), batch_size)
                acc_meter.update(acc.detach().numpy(), batch_size)

            if batch_id > 0 and batch_id % report_freq == 0:
                print(f'----- Batch[{batch_id}/{len(dataloader)}], Loss: {loss_meter.avg:.4}, Acc: {acc_meter.avg:.4}')

    print(f'----- Validation Loss: {loss_meter.avg:.4}, Acc@1: {acc_meter.avg:.4}')


def train():
    total_epoch = 50
    batch_size = 128  # 调试时可先用小一点的，比如16，训练512较好

    # 1. model; 2. dataset; 3. loss; 4. optimizer; 5. learning rate scheduler (optional).

    model = ResNet18(num_classes=10)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_dataset = get_dataset(train=True)
    train_dataloader = get_dataloader(train_dataset, batch_size=batch_size, mode='train')

    val_dataset = get_dataset(train=False)
    val_dataloader = get_dataloader(val_dataset, batch_size=batch_size, mode='test')

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)

    # schedule learning rate.
    # PyTorch是先定义optimizer，再通过一个scheduler来改optimizer里的参数
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch)
    # For CosineAnnealingLR, Pytorch implementation does not have initial lr setting, while paddle implementation has.

    for epoch in range(1, total_epoch+1):
        # time_1 = time.time()
        train_one_epoch(device, model, train_dataloader, criterion, optimizer, epoch, total_epoch)
        scheduler.step() # change learning rate after each epoch

        # time_2 = time.time()
        validate(device, model, val_dataloader, criterion)
        # time_3 = time.time()

        # print('Training time: ', time_2 - time_1)
        # print('Validation time: ', time_3 - time_2)



if __name__ == "__main__":
    train()

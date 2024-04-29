import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


def train_epoch(model, optimizer, loader):
    """
    :param model: (nn.Module type class), should be loaded to GPU 
    :param optimizer: optimizer (torch lib)
    :param loader: data loader (torch lib)
    :return: loss, acc
    """
    model.train()
    train_loss = 0
    train_acc = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        batch_loss = F.cross_entropy(output, target)
        batch_loss.backward()
        optimizer.step()
        train_loss += batch_loss.data[0]
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()

    train_loss /= len(loader)
    train_acc /= len(loader.dataset)
    return train_loss, train_acc


def eval_epoch(model, loader):
    model.eval()
    test_loss = 0
    test_acc = 0

    for data, target in loader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = model(data)
        test_loss += F.cross_entropy(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        test_acc += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(loader)
    test_acc /= len(loader.dataset)
    return test_loss, test_acc


def train_vanilla(
        model, optimizer, train_loader, test_loader,
        model_path, exp_name, log_path = None,
        lr = 0.0001, batch_size = 32, n_epoch = 30):

    log_loss = []
    log_acc = []
    for e in range(n_epoch):
        t_begin = time.time()
        print('---Epoch : ', e+1)
        train_loss, train_acc = train_epoch(model, optimizer, train_loader)
        test_loss, test_acc = eval_epoch(model, test_loader)
        t_end = time.time()

        # epoch summary
        print('train loss : {:.6f}\t|\ttrain acc : {:.6f}'.format(train_loss, train_acc))
        print('test loss : {:.6f}\t|\ttest acc : {:.6f}'.format(test_loss, test_acc))
        print('time : ', t_end - t_begin, ' (s)')

        #stack log
        log_loss.append(train_loss)
        log_acc.append(train_acc)

    torch.save(model.state_dict, model_path + exp_name)
    log_dict = {'loss' : log_loss, 'acc' : log_acc}
    if log_path is not None:
        np.save(log_path + exp_name + '_log.npy', log_dict)


def train_lr_per_epoch(
        model, optimizer, train_loader, test_loader,
        model_path, exp_name, log_path = None,
        lr = None, batch_size = 32, n_epoch = 30):

    if lr is None:
        print('lr should be list or tuple with the same length of n_epoch')
        raise NotImplementedError
    elif len(lr) != n_epoch:
        print('lr should be list or tuple with the same length of n_epoch')
        raise NotImplementedError


    log_loss = []
    log_acc = []
    for e in range(n_epoch):
        for pg in optimizer.param_groups:
            pg['lr'] = lr[e]
        t_begin = time.time()
        print('---Epoch : ', e+1)
        print('lr : ', lr[e])
        train_loss, train_acc = train_epoch(model, optimizer, train_loader)
        if (e+1) % 10 == 0:
            test_loss, test_acc = eval_epoch(model, test_loader)
        t_end = time.time()

        # epoch summary
        print('train loss : {:.6f}\t|\ttrain acc : {:.6f}'.format(train_loss, train_acc))
        if (e+1) % 10 == 0:
            print('test loss : {:.6f}\t|\ttest acc : {:.6f}'.format(test_loss, test_acc))
        print('time : ', t_end - t_begin, ' (s)')

        #stack log
        log_loss.append(train_loss)
        log_acc.append(train_acc)

    torch.save(model.state_dict, model_path + exp_name + '.pt')
    log_dict = {'loss' : log_loss, 'acc' : log_acc}
    if log_path is not None:
        np.save(log_path + exp_name + '_log.npy', log_dict)


def train_with_early_stopping(
        model, optimizer, train_loader, valid_loader, test_loader,
        model_path,
        initial_lr = 0.01, batch_size = 32, max_epoch = 100,
        decay_time = 3, decay_factor = 0.1, max_patience = 3):

    return 0
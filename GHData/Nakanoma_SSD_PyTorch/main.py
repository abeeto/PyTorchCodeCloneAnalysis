from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse


# If you resume training, set weight-file to args['resume']
args = {'dataset': 'VOC',
        'basenet': 'vgg16_reducedfc.pth',
        'batch_size': 8,
        'resume': '',
        'start_iter': 0,
        'num_workers': 4,
        'cuda': False,
        'lr': 5e-4,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'save_folder': 'weights/'
        }


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args['lr'] * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if __name__ == '__main__':
    # set GPU Tensor as a default in creating Tensor
    if torch.cuda.is_available():
        if args['cuda']:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args['cuda']:
            print("WARNING: It looks like you have a CUDA device, but aren't " +
                  "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # create train dataset
    cfg = voc
    dataset = VOCDetection(root=VOC_ROOT,
                           transform=SSDAugmentation(cfg['min_dim'], MEANS))

    # define network
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    device = 'cuda' if torch.cuda.is_available() and args['cuda'] else 'cpu'
    net = ssd_net.to(device)

    # If resume training, load weights of args['resume']
    if args['resume']:
        print('Resuming training, loading {}...'.format(args['resume']))
        ssd_net.load_weights(args['save_folder'] + args['resume'])
    # If train from scratch, load weights of args['basenet']
    else:
        vgg_weights = torch.load(args['save_folder'] + args['basenet'])
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    # output ModuleList of Network
    print(net)

    if args['cuda']:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    # If train from scratch, initialize filters of extra_layer, loc_layers, and conf_layers
    if not args['resume']:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    # set criterion
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args['cuda'])

    # set optimizer
    optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=args['momentum'],
                          weight_decay=args['weight_decay'])

    # train mode
    net.train()
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args['batch_size']
    print(epoch_size)
    print('dataset_size', len(dataset))
    print('epoch_size', epoch_size)
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    # create DataLoader from train dataset
    data_loader = data.DataLoader(dataset, args['batch_size'],
                                  num_workers=args['num_workers'],
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    # start training
    batch_iterator = None

    for iteration in range(args['start_iter'], cfg['max_iter']):

        # 学習開始時または1epoch終了後にdata_loaderから訓練データをロードする
        if (batch_iterator is None) or (iteration % epoch_size == 0):
            print('hello12')
            batch_iterator = iter(data_loader)
            print('hello13')
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args['gamma'], step_index)

        # load train data
        # バッチサイズ分の訓練データをload
        images, targets = next(batch_iterator)

        # 画像をGPUに転送
        images = images.to(device)
        # アノテーションをGPUに転送
        targets = [ann.to(device) for ann in targets]
        # forward
        t0 = time.time()
        # 順伝播の計算
        out = net(images)
        # 勾配の初期化
        optimizer.zero_grad()
        # 損失関数の計算
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        # 勾配の計算
        loss.backward()
        # パラメータの更新
        optimizer.step()
        t1 = time.time()
        # 損失関数の更新
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        # ログの出力
        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

    # 学習済みモデルの保存
    torch.save(ssd_net.state_dict(),
               args['save_folder'] + '' + args['dataset'] + '.pth')
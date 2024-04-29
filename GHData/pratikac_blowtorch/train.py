import torch as th
import torch.optim as optim

import torch.nn as nn
import torchnet as tnt

from timeit import default_timer as timer

import numpy as np
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from pprint import pprint
import  os, pdb, sys, json, subprocess, \
        argparse, math, copy, glob
from copy import deepcopy
from scipy import interpolate

from blowtorch import models, loss, exptutils, loader, viz

opt = exptutils.add_args([
['-o', '/data/results', 'output'],
['-m', 'lenet', 'lenet | mnistfc | allcnn | wrn* | resnet*'],
['--frac', 1., 'dataset fraction'],
['--dataset', 'mnist', 'mnist | cifar10 | cifar100 | imagenet | svhn*'],
['-b', 128, 'bsz'],
['-B', 100, '#epochs'],
['-j', 1, '#gpus'],
['--lr', 0.1, 'lr'],
['--lrs', '[[0,0.1]]', 'lr schedule'],
['-s', 42, 'seed'],
['--env', '', 'visdom env'],
['-v', False, 'verbose'],
['--freq', 1, 'val/save freq.'],
['--save', False, 'save'],
['-r', '', 'reload model']
])
opt['augment'] = True

def train(e, model, criterion, optimizer):
    model.train()
    lr = exptutils.schedule(e, opt, 'lr')
    exptutils.set_lr(optimizer, lr)

    g = opt['g']
    ds = loader.get_loader(opt, is_train=True)

    maxb = len(ds)
    ts, ts2 = timer(), timer()
    s = dict(lr=lr, e=e, f=[], top1=[])
    for bi, (x,y) in enumerate(ds):
        x, y = x.to(g), y.to(g)
        model.zero_grad()
        yh = model(x)
        f = criterion(yh, y)
        f.backward()

        optimizer.step()

        s['f'].append(f.item())
        s['top1'].append(exptutils.error(yh, y))

        if timer() - ts2 > 5:
            print((exptutils.color('blue', '[%2d][%4d/%4d] %2.4f %.2f%%'))%(e,bi,maxb,
                    np.mean(s['f']), np.mean(s['top1'])))
            ts2 = timer()

    print((exptutils.color('blue', '+[%2d] %2.4f %2.2f%% [%.2fs]'))% (e,
        np.mean(s['f']), np.mean(s['top1']), timer()-ts))

    return s

def val(e, model, criterion):
    model.eval()

    g = opt['g']
    ds = loader.get_loader(opt, is_train=False)

    maxb = len(ds)
    ts, ts2 = timer(), timer()
    s = dict(e=e, f=[], top1=[])

    with th.no_grad():
        for bi, (x,y) in enumerate(ds):
            x, y = x.to(g), y.to(g)
            yh = model(x)
            f = criterion(yh, y)

            s['f'].append(f.item())
            s['top1'].append(exptutils.error(yh, y))

            if timer() - ts2 > 5:
                print((exptutils.color('red', '[%2d][%4d/%4d] %2.4f %.2f%%'))%(e,bi,maxb,
                        np.mean(s['f']), np.mean(s['top1'])))
                ts2 = timer()

    print((exptutils.color('red', '*[%2d] %2.4f %2.2f%% [%.2fs]'))% (e,
        np.mean(s['f']), np.mean(s['top1']), timer()-ts))

    return s

def reload(model):
    if opt['r'] == '':
        return 0, [], []

    assert os.path.exists(opt['r']), 'Could not find: %s'%opt['r']

    def check_opt(o):
        global opt
        print('Old opt: ')
        pprint(opt)
        print('New opt: ')
        pprint(o)
        r = input('Press y[yes] to continue: ')
        if r == 'y' or r == 'yes':
            opt = deepcopy(o)

    d = th.load(opt['r'])
    model.load_state_dict(d['state_dict'])
    check_opt(d['opt'])
    return d['e']+1, d['train_stats'], d['val_stats']

def save(d):
    if not opt['save']:
        return

    fn = os.path.join(opt['o'], opt['fname'] + '.pt')
    th.save(d, fn)

def setup():
    # setup rand and gpus
    exptutils.setup(opt)

    # logging/saving
    blklist = ['augment', 'b', 'd', 'frac', 'g', 'gs', 'env', 'freq', 'meta',
                'j', 'lr', 'lrs', 's', 'save', 'v', 'l2','depth','widen']

    if not 'fname' in opt:
        exptutils.build_filename(opt, blklist)

    # git status
    opt['meta'] = exptutils.gitrev(opt)

    # visdom
    s = opt['fname']
    opt['title'] = s[s.find("{")+1:s.find("}")]

def main():
    model = getattr(models, opt['m'])(opt)
    criterion = loss.wrap(nn.CrossEntropyLoss(), loss.ell2(opt, model))

    lr = exptutils.schedule(0, opt, 'lr')
    optimizer = th.optim.SGD(model.parameters(), lr=lr,
                             nesterov=True, momentum=0.9)

    start_e, sts, svs = reload(model)
    model, criterion = exptutils.cudafy(opt, model, criterion)

    pprint(opt)
    for e in range(start_e, opt['B']):
        print('')
        st, sv = None, None

        st = train(e, model, criterion, optimizer)
        sts.append(st)

        if e % opt['freq'] == 0 or e == opt['B']-1:
            sv = val(e, model, criterion)
            svs.append(sv)

            save(dict(
                      opt=opt,
                      e=e, train_stats=sts, val_stats=svs,
                      state_dict=model.state_dict())
            )

setup()
main()
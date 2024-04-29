import numpy as np
import argparse
import time
import torch
import os
import warnings
import torch.nn as nn

from procrustes import *
from model import *
from cameras import *
from config import *
from utils import *
from save_utils import *
from dataloader import *

warnings.filterwarnings("ignore")

def lr_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step / decay_step)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr

def train(opts, model, rcams, criterion, optimizer):
    if opts.load:
        ckpt = torch.load(CKPT_PATH)
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        global_step = ckpt['step']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
    
    if opts.resume:
        logger = Logger(os.path.join(LOG_PATH, 'log.txt'), resume=True)
    else:
        logger = Logger(os.path.join(LOG_PATH, 'log.txt'))
        logger.set_names(['epoch', 'lr', 'loss_train', 'loss_test', 'err_test'])
    
    global_step = 0
    err_best = 1000
    lr_init = opts.lr
    lr_now = opts.lr
    
    train_dataset = Human36m(data_path=BASE_PATH, data_path_sh='', actions=ACTIONS, rcams=rcams, use_sh=False, training=True)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataset = Human36m(data_path=BASE_PATH, data_path_sh='', actions=ACTIONS, rcams=rcams, use_sh=False, training=False)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
    torch.backends.cudnn.benchmark = True
    
    for epoch in range(opts.epochs):
        losses = AverageMeter()
        step_start = epoch_start = time.time()
        step_time = 0
        
        model.train()
        
        print('Epoch: {}, lr: {:.5f}\n'.format(epoch + 1, lr_now))
        
        for idx, (x, y) in enumerate(train_dataloader):
            global_step += 1
            
            if global_step % opts.lr_decay == 0 or global_step == 1:
                lr_now = lr_decay(optimizer, global_step, lr_init, opts.lr_decay, opts.lr_gamma)
            
            inputs = torch.autograd.Variable(x.to(device))
            targets = torch.autograd.Variable(y.to(device))
            
            outputs = model(inputs)
            
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            loss.backward()
            
            if opts.max_norm:
                nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
                
            optimizer.step()
            
            if (idx + 1) % 100 == 0:
                step_time = time.time() - step_start
                step_start = time.time()
                
            print('Training: Batch: {}/{}, 100 Steps Time: {}, Loss: {:.4f}'.format(idx + 1, len(train_dataloader), step_time, losses.avg), end='\r')
        print('\nTime for 1 epoch: {}'.format(time.time() - epoch_start))
            
        loss_test, err_test, _ = evaluate(opts, model, rcams, test_dataset, test_dataloader, procrustes=opts.procrustes)
        
        logger.append([epoch + 1, lr_now, losses.avg, loss_test, err_test], ['int', 'float', 'float', 'float', 'float'])
        
        is_best = err_test < err_best
        err_best = min(err_test, err_best)
        
        save_dict = {
                'epoch': epoch + 1,
                'lr': lr_now,
                'step': global_step,
                'err': err_best,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
        
        if is_best:
            save_ckpt(save_dict, save_path=CKPT_PATH, is_best=True)
        else:
            save_ckpt(save_dict, save_path=CKPT_PATH, is_best=False)
            
    logger.close()
    
def evaluate(opts, model, rcams, test_dataset, test_dataloader, procrustes=False):         
    losses = AverageMeter()
    
    model.eval()
    
    all_dist = []
    start = time.time()
    step_time = 0
    
    # data_3d_mean, data_3d_std = test_dataset.test_y_mean, test_dataset.test_y_std
    # data_dim_to_use = test_dataset.test_y_dim_to_use
    data_3d_mean, data_3d_std, data_dim_to_use, _ = read_stats_data(dim=3)
    
    for idx, (x, y) in enumerate(test_dataloader):
        inputs = torch.autograd.Variable(x.to(device))
        targets = torch.autograd.Variable(y.to(device))
        
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        
        dim_use = np.hstack((np.arange(3), data_dim_to_use))
        
        origin_targets = unNormalize_data(targets.data.cpu().numpy(), data_3d_mean, data_3d_std, data_dim_to_use)[:, dim_use]
        origin_outputs = unNormalize_data(outputs.data.cpu().numpy(), data_3d_mean, data_3d_std, data_dim_to_use)[:, dim_use]
        
        if procrustes:
            for j in range(inputs.size(0)): # Sometimes 32, basically 64
                gt = origin_targets[j].reshape(-1, 3)
                out = origin_outputs[j].reshape(-1, 3)
                _, Z, T, b, c = compute_similarity_transform(gt, out, complete_optimal_scale=True)
                out = (b * out.dot(T)) + c
                origin_outputs[j, :] = out.reshape(-1, 17 * 3)
        
        square_err = (origin_outputs - origin_targets) ** 2
        
        distance = np.zeros((square_err.shape[0], 17))
        dist_idx = 0
        for i in np.arange(0, 17 * 3, 3):
            distance[:, dist_idx] = np.sqrt(np.sum(square_err[:, i: i + 3], axis=1))
            dist_idx += 1
        all_dist.append(distance)
        
        if (idx + 1) % 100 == 0:
            step_time = time.time() - start
            start = time.time()
        
        print('\n')
        print('Testing: Batch: {}/{}, 100 Steps Time: {}, Loss: {:.4f}'.format(idx + 1, len(test_dataloader), step_time, losses.avg), end='\r')
        
    all_dist = np.vstack(all_dist)
    joint_err = np.mean(all_dist, axis=0)
    total_err = np.mean(all_dist)
    
    return losses.avg, total_err, joint_err

def test(opts, model, rcams, criterion):
    ckpt = torch.load(opts.ckpt_dir)
    model.load_state_dict(ckpt['state_dict'])
    
    err_set = []
        
    for action in ACTIONS:
        print('Test on {}'.format(action))
        test_dataset = Human36m(data_path=BASE_PATH, data_path_sh='', actions=action, rcams=rcams, use_sh=False, training=False)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        _, err_test, _ = evaluate(opts, model, rcams, test_dataset, test_dataloader, procrustes=opts.procrustes)
        err_set.append(err_test)
        
    for action in ACTIONS:
        print ("{}".format(action), end='\t')
    print ("\n")
    for err in err_set:
        print ("{:.4f}".format(err), end='\t')
    print ("\nERRORS: {}".format(np.array(err_set).mean()))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PoseBaseline')
    
    parser.add_argument('--training', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=int, default=100000)
    parser.add_argument('--lr_gamma', type=float, default=0.96)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_norm', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--load', action='store_true', default=False)
    parser.add_argument('--procrustes', action='store_true', default=False)
    parser.add_argument('--ckpt_dir', type=str, default='ckpt/ckpt_best.pt')
    opts = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rcams = cameras.load_cameras()
    
    model = LinearModel()
    model = model.cuda()
    model.apply(weights_init)
    
    criterion = nn.MSELoss(size_average=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    
    if opts.training:
        print('Training')
        train(opts, model, rcams, criterion, optimizer)
    else:
        print('Testing')
        test(opts, model, rcams, criterion)

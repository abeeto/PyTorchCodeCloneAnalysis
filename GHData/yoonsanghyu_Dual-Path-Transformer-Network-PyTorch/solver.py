# Created on 2018/12
# Author: Kaituo XU
# Edited by yoonsanghyu

import os
import time
import numpy as np
import torch

from pit_criterion import cal_loss

class TransformerOptimizer(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, k, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.k = k
        self.init_lr = d_model ** (-0.5)
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.epoch = 0
        self.visdom_lr = None

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, epoch):
        self._update_lr(epoch)
        # self._visdom()
        self.optimizer.step()

    def _update_lr(self, epoch):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.k * self.init_lr * min(self.step_num ** (-0.5),
                                             self.step_num * (self.warmup_steps ** (-1.5)))
        else:
            lr = 0.0004 * (0.98 ** ((epoch-1)//2))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def set_k(self, k):
        self.k = k

    def set_visdom(self, visdom_lr, vis):
        self.visdom_lr = visdom_lr  # Turn on/off visdom of learning rate
        self.vis = vis  # visdom enviroment
        self.vis_opts = dict(title='Learning Rate',
                             ylabel='Leanring Rate', xlabel='step')
        self.vis_window = None
        self.x_axis = torch.LongTensor()
        self.y_axis = torch.FloatTensor()

    def _visdom(self):
        if self.visdom_lr is not None:
            self.x_axis = torch.cat(
                [self.x_axis, torch.LongTensor([self.step_num])])
            self.y_axis = torch.cat(
                [self.y_axis, torch.FloatTensor([self.optimizer.param_groups[0]['lr']])])
            if self.vis_window is None:
                self.vis_window = self.vis.line(X=self.x_axis, Y=self.y_axis,
                                                opts=self.vis_opts)
            else:
                self.vis.line(X=self.x_axis, Y=self.y_axis, win=self.vis_window,
                              update='replace')


class Solver(object):
    
    def __init__(self, data, model, optimizer, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.optimizer =  TransformerOptimizer(optimizer, 0.2, 64)

        # Training config
        self.use_cuda = args.use_cuda
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.max_norm = args.max_norm
        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.model_path = args.model_path
        # logging
        self.print_freq = args.print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self.visdom = args.visdom
        self.visdom_epoch = args.visdom_epoch
        self.visdom_id = args.visdom_id
        if self.visdom:
            from visdom import Visdom
            self.vis = Visdom(env=self.visdom_id)
            self.vis_opts = dict(title=self.visdom_id,
                                 ylabel='Loss', xlabel='Epoch',
                                 legend=['train loss', 'cv loss'])
            self.vis_window = None
            self.vis_epochs = torch.arange(1, self.epochs + 1)

        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            cont = torch.load(self.continue_from)
            self.start_epoch = cont['epoch']
            self.model.load_state_dict(cont['model_state_dict'])
            self.optimizer.load_state_dict(cont['optimizer_state'])
            torch.set_rng_state(cont['trandom_state'])
            np.random.set_state(cont['nrandom_state'])

        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False
        self.val_no_impv = 0

    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            print("Training...")
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)

            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'trandom_state': torch.get_rng_state(),
                    'nrandom_state': np.random.get_state()}, file_path)
                print('Saving checkpoint model to %s' % file_path)

            # Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            with torch.no_grad():
                val_loss = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, val_loss))
            print('-' * 85)

            # Adjust learning rate (halving)
            if self.half_lr:
                if val_loss >= self.prev_val_loss:
                    self.val_no_impv += 1
                    if self.val_no_impv >= 3:
                        self.halving = True
                    if self.val_no_impv >= 10 and self.early_stop:
                        print("No imporvement for 10 epochs, early stopping.")
                        break
                else:
                    self.val_no_impv = 0
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] / 2.0
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                self.halving = False
            self.prev_val_loss = val_loss

            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_file_path = os.path.join(
                    self.save_folder, 'temp_best.pth.tar')
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'trandom_state': torch.get_rng_state(),
                    'nrandom_state': np.random.get_state()}, best_file_path)
                print("Find better validated model, saving to %s" % best_file_path)

            # visualizing loss using visdom
            if self.visdom:
                x_axis = self.vis_epochs[0:epoch + 1]
                y_axis = torch.stack(
                    (self.tr_loss[0:epoch + 1], self.cv_loss[0:epoch + 1]), dim=1)
                if self.vis_window is None:
                    self.vis_window = self.vis.line(
                        X=x_axis,
                        Y=y_axis,
                        opts=self.vis_opts,
                    )
                else:
                    self.vis.line(
                        X=x_axis.unsqueeze(0).expand(y_axis.size(
                            1), x_axis.size(0)).transpose(0, 1),  # Visdom fix
                        Y=y_axis,
                        win=self.vis_window,
                        update='replace',
                    )

    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        # visualizing loss using visdom
        if self.visdom_epoch and not cross_valid:
            vis_opts_epoch = dict(title=self.visdom_id + " epoch " + str(epoch),
                                  ylabel='Loss', xlabel='Epoch')
            vis_window_epoch = None
            vis_iters = torch.arange(1, len(data_loader) + 1)
            vis_iters_loss = torch.Tensor(len(data_loader))

        for i, (data) in enumerate(data_loader):
            padded_mixture, mixture_lengths, padded_source = data
            if self.use_cuda:
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()
            estimate_source = self.model(padded_mixture)
            loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_norm)
                self.optimizer.step(epoch)

            total_loss += loss.item()

            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)),
                      flush=True)

            # visualizing loss using visdom
            if self.visdom_epoch and not cross_valid:
                vis_iters_loss[i] = loss.item()
                if i % self.print_freq == 0:
                    x_axis = vis_iters[:i+1]
                    y_axis = vis_iters_loss[:i+1]
                    if vis_window_epoch is None:
                        vis_window_epoch = self.vis.line(X=x_axis, Y=y_axis,
                                                         opts=vis_opts_epoch)
                    else:
                        self.vis.line(X=x_axis, Y=y_axis, win=vis_window_epoch,
                                      update='replace')

        return total_loss / (i + 1)
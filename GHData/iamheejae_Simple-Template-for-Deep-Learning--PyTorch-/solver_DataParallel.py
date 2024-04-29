from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import DataParallel as DP
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
summary=SummaryWriter('./myruns/log')

import math
import os
import time
import sys
import numpy as np

sys.path.insert(0,'./models/')
sys.path.insert(0,'./utils/')

import pdb
import cv2

class Solver(object):
    
    def __init__(self, model, check_point, **kwargs):
        """
        Construct a new Solver instance

        """
        self.model = model
        self.check_point = check_point
        self.num_epochs = kwargs.pop('num_epochs')
        self.batch_size = kwargs.pop('batch_size')
        self.learning_rate = kwargs.pop('learning_rate')
        self.optimizer = kwargs.pop('optimizer')
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=1)
        self.fine_tune = kwargs.pop('fine_tune', False)        
        self.verbose = kwargs.pop('verbose', False)
        self.gpus = kwargs.pop('gpus')
        self.parallel = kwargs.pop('parallel')
        self._reset()

        #************************* Losse Function *************************#
        self.loss_fn = torch.nn.MSELoss()  # Define your loss function

    def _reset(self):

        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:            
            if self.parallel:
                #---------Train with DataParallel(Single Machine)---------#                
                self.model = self.model.cuda()
                self.model = DP(self.model)
            else: 
                self.model = self.model.cuda()

    def _epoch_step(self, dataset, epoch):        

        """ Perform one 'epoch' on the training dataset"""
        self.model.train() 
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.gpus)           
        
        num_iters = len(dataset)//self.batch_size
        running_loss= 0


        for i, (GT, img_t, img_s) in enumerate(tqdm(dataloader)):            

            B, C, H, W = img_t.size()

            # Convert input to torch variable
            img_t, img_s = self._wrap_variable(img_t, img_s, self.use_gpu)

            self.optimizer.zero_grad()

            # Forward     
            net_output = self.model(img_t, img_s)            
            pred = net_output['out']                            

            loss = self.loss_fn(pred, GT.to(device = pred.device))          
            running_loss += loss.item()
            
            # Backward and update
            loss.backward()
            self.optimizer.step()

        average_loss = running_loss/num_iter    

        if self.verbose:
            print('Epoch  %5d, loss %.5f' % (epoch, average_loss))
        return average_loss

    def _wrap_variable(self, input1, input2, use_gpu):

        if use_gpu:
            input1, input2 = (Variable(input1.cuda()), Variable(input2.cuda()))

        else:
            input1, input2 = (Variable(input1), Variable(input2))
        return input1, input2


    def _check_val(self, dataset, is_test=False):

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.gpus)
        self.model.eval()

        avr_val_loss = 0
        
        for batch, (GT, img_t, img_s) in enumerate(dataloader):
            img_t, img_s = self._wrap_variable(img_t, img_s, self.use_gpu)

            if is_test:
                start = time.time()
                net_output = self.model(img_t, img_s)
                output_batch = net_output['out']
                elapsed_time = time.time() - start
            else:
                net_output = self.model(img_t, img_s)
                output_batch = net_output['out']
            
            output = output_batch.cuda().data
            label = GT.cuda()
            val_loss = self.loss_fn(output, label)
            avr_val_loss += val_loss.item()

        epoch_size = len(dataset)
        avr_val_loss /= epoch_size
        stats = val_loss

        return avr_val_loss, stats, output.cpu().numpy()

    def train(self, train_dataset, val_dataset):

        # Check fine_tuning option
        fine_tune_path = os.path.join(self.check_point, 'model_fine_tune.pt')
        if self.fine_tune and not os.path.exists(fine_tune_path):
            raise Exception('Cannot find %s.' % fine_tune_path)

        elif self.fine_tune and os.path.exists(fine_tune_path):
            if self.verbose:
                print('Loading %s for finetuning.' % fine_tune_path)
            
            ckpt = torch.load(fine_tune_path)
            self.model.module.load_state_dict(ckpt['model_state_dict'])

               
        # Capture best model
        best_val_loss = -1

        # Train the model       
        for epoch in range(self.num_epochs):
            curr_epoch_loss = self._epoch_step(train_dataset, epoch)                  
             
            summary.add_scalar('Train/Total Loss per Epoch', curr_epoch_loss, epoch)
            self.scheduler.step()

            # Validation 
            if self.verbose:
                print('Validation with current epoch')
            val_loss, _, val_out = self._check_val(val_dataset)

            if self.verbose:
                print('Val loss: %.3f.'% (val_loss))
            summary.add_scalar('Validation/loss per Epoch', val_loss, epoch)


            # Checkpoint file directory
            print('Saving model')
            if not os.path.exists(self.check_point):
                os.makedirs(self.check_point)
            model_path = os.path.join(self.check_point, 'epoch{}.pt'.format(epoch))
          
            if self.parallel:
                torch.save({'epoch': epoch, 'model_state_dict': self.model.module.state_dict(), 'loss':curr_epoch_loss}, model_path)
            else:
                torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'loss':curr_epoch_loss, model_path)

            if best_val_loss < val_loss:
                print('Copy best model')
                best_path = os.path.join(self.check_point, 'best_model.pt')
                copyfile(model_path, best_path)
                best_val_loss = val_loss
            print('')

        summary.close()

    def test(self, dataset, model_path):
        if not os.path.exists(model_path):
            raise Exception('Cannot find %s.' % model_path)

        _, stats, outputs = self._check_val(dataset, is_test=True)
        return stats, outputs



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
# Created on 27 21:27

"""Trainer"""

from datasets.alov import ALOV300
from datasets.det import DET2014
from datasets.vot import VOT2015
from datasets.vid import VID2015
from datasets.example_generator import Example_generator
from configuration import DATA_CONFIG, TRAIN_CONFIG, MODEL_CONFIG, RUN_NAME, CONF_FN, bcolors
from datasets.dataloader import PairData, StaticPairData
from torch.utils.data import DataLoader
from multiprocessing import Queue, Process
from logger.logger import setup_logger
import importlib
import torch
import os
import sys
import os.path as osp
from utils.misc_utils import mkdir_p, bboxes_aveiou
from utils.train_utils import BatchLoader, StepLR
import torch.optim as optim
from torch.autograd import Variable
#from torch.optim.lr_scheduler import MultiStepLR, StepLR
import logging
import cProfile
import pstats
import numpy as np
np.random.seed(TRAIN_CONFIG['seed'])

class Trainer:
    def __init__(self, logger):
        self.bs = TRAIN_CONFIG['bs']
        self.logger = logging if logger is None else logger
        self.dloader = self.get_dloader()
        self.logger.info("dloader set")
        self.from_iter, self.model = self.get_model()
        self.logger.info("model set")
        self.criterion = self.get_criterion()
        self.logger.info("criterion set")
        self.cuda()     # Must before optimizer
        self.optimizer, self.scheduler = self.get_optimizer_scheduler()
        self.logger.info("optimizer and cuda set")

        #self.batchfeeder = BatchFeeder(self.dataloader[0], self.dataloader[1], MODEL_CONFIG['input_size'], TRAIN_CONFIG['bs'], TRAIN_CONFIG['kGenPerImage'])

    def get_dloader(self):
        self.exgenerator = Example_generator(TRAIN_CONFIG['motionparam'])
        print (DATA_CONFIG['alov_vdroot'], DATA_CONFIG['alov_adroot'])
        alov_videos = ALOV300(DATA_CONFIG['alov_vdroot'], DATA_CONFIG['alov_adroot'], self.logger)
        alov_obj = PairData(alov_videos)
        #alov_loader = DataLoader(alov_obj, batch_size=TRAIN_CONFIG['bs'], shuffle=TRAIN_CONFIG['shuffle'],
        #                         pin_memory=True, num_workers=TRAIN_CONFIG['num_workers'])

        det_images = DET2014(DATA_CONFIG['det2014_imroot'], DATA_CONFIG['det2014_adroot'])
        det_obj = StaticPairData(det_images.all_impath, det_images.all_adpath, DATA_CONFIG['det_dbfn'])
        #det_loader = DataLoader(det_obj, batch_size=TRAIN_CONFIG['bs'], shuffle=TRAIN_CONFIG['shuffle'],
        #                        pin_memory=True, num_workers=TRAIN_CONFIG['num_workers'])
        #vid_videos = VID2015(DATA_CONFIG['vid2014_vdroot'], DATA_CONFIG['vid2014_adroot'], DATA_CONFIG['vid2014_dbfn'])
        #vid_obj = PairData(vid_videos, DATA_CONFIG['vid_wrapper_dbfn'])
        #vid_loader = DataLoader(vid_obj, batch_size=TRAIN_CONFIG['bs'], shuffle=TRAIN_CONFIG['shuffle'],
        #                         num_workers=TRAIN_CONFIG['num_workers'])
        dloader = BatchLoader(alov_obj, det_obj, self.exgenerator, 
                                      MODEL_CONFIG['input_size'], 
                                      self.bs, 
                                      TRAIN_CONFIG['kGenPerImage'])
        return dloader
        #from datasets.raw import RawPair
        #loader = RawPair("/data/qiang.zhou/RAWDET2014", "/data/qiang.zhou/RAWDET2014/label.txt")
        #return loader

    def get_model(self):
        model = importlib.import_module("models." + MODEL_CONFIG['model_id']).GONET(MODEL_CONFIG['init_model_dir'])
        resume_checkpoint = TRAIN_CONFIG['resume_checkpoint']
        if resume_checkpoint is None:
            from_iter = 0
        else:
            from_iter = int(resume_checkpoint.split('.')[0].split('_')[-1])    # Checkpoint should in Epoch_10.pth.tar
            if os.path.exists(resume_checkpoint):
                if TRAIN_CONFIG['use_gpu']:
                    model.load_state_dict(torch.load(resume_checkpoint))
                else:
                    model.load_state_dict(torch.load(resume_checkpoint, map_location=lambda storage, loc: storage))
            else:
                assert False, "The trained model state file not exists..."
        self.logger.info("model {} from iteration {}".format(MODEL_CONFIG['model_id'], from_iter))
        return from_iter, model

    def get_criterion(self):
        return torch.nn.L1Loss(size_average=MODEL_CONFIG['size_average'])
        #return torch.nn.MSELoss(size_average=MODEL_CONFIG['size_average'])

    def get_optimizer_scheduler(self):
        layer_scheme = []
        w_decay = TRAIN_CONFIG['weight_decay']
        lr = TRAIN_CONFIG['init_lr']
        for name, param in self.model.named_parameters():
            if 'features' in name and 'weight' in name:
                layer_scheme.append({'params': param, 'lr': 0, 'weight_decay': w_decay})
            elif 'features' in name and 'bias' in name:
                layer_scheme.append({'params': param, 'lr': 0, 'weight_decay': 0      })
            elif 'regressor' in name and 'weight' in name:
                layer_scheme.append({'params': param,'lr':lr*10,'weight_decay':w_decay})
            elif 'regressor' in name and 'bias' in name:
                layer_scheme.append({'params': param,'lr':lr*20,'weight_decay':0      })
            else:
                self.logger.warning('Layer {} not idientifed...'.format(name))
                
        optimizer = optim.SGD(layer_scheme, momentum=TRAIN_CONFIG['momentum'])
        #scheduler = MultiStepLR(optimizer, milestones=TRAIN_CONFIG['milestones'], gamma=TRAIN_CONFIG['lr_gamma'])
        scheduler = StepLR(optimizer, TRAIN_CONFIG['stepsize'], TRAIN_CONFIG['lr_gamma'], self.logger)
        return optimizer, scheduler

    def cuda(self):
        if TRAIN_CONFIG['use_gpu']:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

    def show_batch(self, i_iter, batch_x1, batch_x2, batch_y):
        # Helper function to show a batch
        from torchvision import utils
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import cv2
        batch_size = self.bs
        im_size = batch_x1.size(2)
        grid1 = utils.make_grid(batch_x1).numpy()
        grid2 = utils.make_grid(batch_x2).numpy()

        for i in range(batch_size):
            bb = batch_y[i].numpy() / 10 * im_size
            pt1 = (int(bb[0]+(i%8)*im_size), int(bb[1]+(i//8)*im_size))
            pt2 = (int(bb[2]+(i%8)*im_size), int(bb[3]+(i//8)*im_size))
            print (pt1, pt2)
            cv2.rectangle(grid2, pt1, pt2, (255, 0, 0), 5)

        # WxHxC -> CxWxH            BGR
        mean = np.array([104, 117, 123])
        grid1 = grid1.transpose((1, 2, 0)) + mean
        grid2 = grid2.transpose((1, 2, 0)) + mean
        cv2.imwrite('batch_{:02d}_targets.jpg'.format(i_iter), grid1)
        cv2.imwrite('batch_{:02d}_images.jpg'.format(i_iter), grid2)
        if i_iter >= 10:
            exit()

    def train(self):
        self.model.train()
        n_iter = TRAIN_CONFIG['n_iter']
        best_loss = float('inf')
        best_iou = 0
        
        for i_iter in range(self.from_iter, n_iter):
            batch_x1, batch_x2, batch_y = self.dloader()
            #print ("i_Iter: ", i_iter)
            #self.show_batch(i_iter, batch_x1, batch_x2, batch_y)
            #batch_x1, batch_x2, batch_y = self.dloader.load(i_iter)
            #continue

            if TRAIN_CONFIG['use_gpu']:
                batch_x1, batch_x2, batch_y = Variable(batch_x1.cuda()), Variable(batch_x2.cuda()), \
                                              Variable(batch_y.cuda(), requires_grad=False)
            else:
                batch_x1, batch_x2, batch_y = Variable(batch_x1), Variable(batch_x2), \
                                              Variable(batch_y, requires_grad=False)
            self.optimizer.zero_grad()
            output = self.model(batch_x1, batch_x2)
            loss = self.criterion(output, batch_y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(i_iter)
            self.logger.info('Iter {} -- Loss {}{:.4f}{}'.format(i_iter, bcolors.UNDERLINE, loss.data[0], bcolors.ENDC))
            #aveiou = bboxes_aveiou(output.data.cpu().numpy(), batch_y.data.cpu().numpy())
            #self.logger.info('\tBatch average IOU reaches {}{:.2f}{}'.format(bcolors.OKBLUE, aveiou, bcolors.ENDC))

            # Save model
            if (i_iter % TRAIN_CONFIG['dump_freq'] == 0 and i_iter > 0) or (i_iter==n_iter-1):
                state_fn = osp.join(TRAIN_CONFIG['exp_dir'], "Iteration_{}.pth.tar".format(i_iter))
                torch.save(self.model.state_dict(), state_fn)
                self.logger.info('Model dumped to {}'.format(state_fn))
            
    def valid(self):
        vot_videos = VOT2015(DATA_CONFIG['vot2015_imroot'], DATA_CONFIG['vot2015_dbfn']).videos
        vot_obj = PairData(vot_videos)
        dloader = DataLoader(vot_obj, batch_size=self.bs, shuffle=False, 
                                      pin_memory=True, num_workers=8)
        lossarr = []
        self.model.eval()
        for i, dpair in enumerate(dloader):
            batch_x1, batch_x2, batch_y = dpair['prev_im'], dpair['curr_im'], dpair['curr_bb']
            if TRAIN_CONFIG['use_gpu']:
                batch_x1, batch_x2, batch_y = Variable(batch_x1.cuda()), Variable(batch_x2.cuda()), \
                                              Variable(batch_y.cuda(), requires_grad=False)
            else:
                batch_x1, batch_x2, batch_y = Variable(batch_x1), Variable(batch_x2), \
                                              Variable(batch_y, requires_grad=False)
            output = self.model(batch_x1, batch_x2)
            loss = self.criterion(output, batch_y)
            self.logger.info('Valid: {} Loss {:.4f}'.format(i, loss.data[0]))
            lossarr.append(loss.data[0])
            aveiou = bboxes_aveiou(output.data.cpu().numpy(), batch_y.data.cpu().numpy())
            self.logger.info('\tBatch average IOU reaches {:.2f}'.format(aveiou))
        self.logger.info('VALID loss mean {} var {}'.format(np.mean(lossarr), np.var(lossarr)))


def main():
    exp_dir = TRAIN_CONFIG['exp_dir']
    if not os.path.exists(exp_dir):
        mkdir_p(exp_dir)
    #if len(sys.argv)==2 and sys.argv[1] != '--resume':
    #    assert False, 'Experiment directory {} exists, prohibit overwritting...\n\t**Use python3 train.py --resume to force training...'.format(exp_dir)
    if os.path.exists(TRAIN_CONFIG['logfile']) and TRAIN_CONFIG['resume_checkpoint'] is None:
        os.remove(TRAIN_CONFIG['logfile'])
    logger = setup_logger(logfile=TRAIN_CONFIG['logfile'])
    os.system('cp {} {}'.format(CONF_FN, osp.join(exp_dir, CONF_FN.split('/')[-1])))
    trainer = Trainer(logger)
    trainer.train()


if __name__ == "__main__":
    """
    cProfile.run("main()", "timeit")
    p = pstats.Stats('timeit')
    p.sort_stats('time')
    p.print_stats(6)
    """
    main()

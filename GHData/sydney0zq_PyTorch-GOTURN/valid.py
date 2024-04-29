#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
# Created on 27 21:27

"""Valider"""

from datasets.vot import VOT2015
from datasets.dataloader import PairData
from torch.utils.data import DataLoader
from logger.logger import setup_logger
import importlib
import numpy as np
import torch
import cv2
import os
import sys
import os.path as osp
from utils.misc_utils import mkdir_p, bboxes_aveiou
from torch.autograd import Variable
import logging

bs = 50
votdir = "/data/qiang.zhou/VOT2015"
model_id = "goturn"
pretrained_dir = "models/pretrained"

class Valider:
    def __init__(self, logger):
        self.logger = logging if logger is None else logger
        self.bs = bs
        self.dloader = self.get_dloader()
        self.logger.info("dloader set")
        self.model = self.get_model()
        self.logger.info("model set")
        self.criterion = self.get_criterion()
        self.logger.info("criterion set")
        self.cuda()

    def get_dloader(self):
        #print ("Modify loader to feed your needs")
        #exit()
        vot_videos = VOT2015(votdir).videos
        vot_obj = PairData(vot_videos)
        print ("datalen: {}".format(len(vot_obj)))
        dloader = DataLoader(vot_obj, batch_size=self.bs, shuffle=False,
                                 pin_memory=True, num_workers=20)
        return dloader

    def get_model(self):
        #print ("Modify get model to feed your needs")
        #exit()
        model = importlib.import_module("models." + model_id).GONET(pretrained_dir)
        model.load_state_dict(torch.load("/tmp/Iteration_100000.pth.tar"))
        return model

    def get_criterion(self):
        return torch.nn.L1Loss(size_average=False)

    def cuda(self):
        self.model = self.model.cuda()
        self.criterion = self.criterion.cuda()

    def valid(self):
        # OnePass
        self.model.eval()
        gpu = True
        self.logger.info('start to valid')
        for i, dpair in enumerate(self.dloader):
            batch_x1, batch_x2, batch_y = dpair['prev_im'], dpair['curr_im'], dpair['curr_bb']
            if gpu:
                batch_x1, batch_x2, batch_y = Variable(batch_x1.cuda()), Variable(batch_x2.cuda()), \
                                              Variable(batch_y.cuda(), requires_grad=False)
            else:
                batch_x1, batch_x2, batch_y = Variable(batch_x1), Variable(batch_x2), \
                                              Variable(batch_y, requires_grad=False)
            print ('forward')
            output = self.model(batch_x1, batch_x2)
            loss = self.criterion(output, batch_y)
            self.logger.info('{} Loss {:.4f}'.format(i, loss.data[0]))
            #aveiou = bboxes_aveiou(output.data.cpu().numpy(), batch_y.data.cpu().numpy())
            #self.logger.info('\tBatch average IOU reaches {:.2f}'.format(aveiou))
def main():
    logger = setup_logger(logfile=None)
    valider = Valider(logger)
    valider.valid()


if __name__ == "__main__":
    """
    cProfile.run("main()", "timeit")
    p = pstats.Stats('timeit')
    p.sort_stats('time')
    p.print_stats(6)
    """
    main()


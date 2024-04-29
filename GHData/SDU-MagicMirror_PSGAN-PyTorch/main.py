#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os.path as osp
import time

pwd = osp.split(osp.realpath(__file__))[0]
import sys

sys.path.append(pwd + '/..')

from torch.backends import cudnn
from data_loaders.makeup_utils import *
from solver_psgan import Solver_PSGAN

cudnn.benchmark = True

if __name__ == '__main__':
    images = ['vSYYZ429.png', 'gakki_face.png']
    images = ['vSYYZ429.png', 'gakki_face.png']
    images = [Image.open(image) for image in images]
    images = [preprocess_image(image) for image in images]
    time_total = 0
    start = time.time()
    transferred_image = Solver_PSGAN.image_test(*(images[0]), *(images[1]))
    time_total += time.time() - start
    print('time_total: ', time_total)
    transferred_image.save('429_gakkiface_test.png')

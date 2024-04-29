#-*-coding:utf-8-*-
import sys
import os.path as osp

BASE_DIR = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(BASE_DIR, 'data'))
sys.path.append(osp.join(BASE_DIR, 'model'))
sys.path.append(osp.join(BASE_DIR, 'utils'))

import train
import utils.logger
import utils.augmentation

if __name__ == '__main__' :
    logger = utils.logger.logger()
    try :
        train.train()
    except :
        logger.error("error during train!")
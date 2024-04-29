#!/bin/python3

import torch
import addict
import easydict
import numpy as np

from src.configs import config
from src.api import queue_manager as qm
from src.api import flask
from src.api import redis
from src.modules import dataset as ds
import src.api.response as rs
import src.modules.learner as lrn
from src.modules import smooth_tile_predictions as smt
import src.utils.preprocess as ps
import src.utils.rle as rle
from src.modules import inference
from src.models import regression_tree as rt

import multiprocessing as mp


if __name__ == '__main__':
    mp_queue = mp.Queue()
    r_api = redis.RedisAPI(mp_queue)
    r_api.check()

    redis_process = mp.Process(target=r_api.listen)
    redis_process.start()
    config.SHARED.INIT()

    manager = qm.QueueManager(r_api=r_api, mp_queue=mp_queue)

    while True:
        manager.start()

    tracebacks = list()
    while True:
        try:
            clear_output(wait=True)
            manager.start()    
        except Exception:
            traceback.print_exc(file=sys.stdout)
            tracebacks.append(traceback.format_exc())

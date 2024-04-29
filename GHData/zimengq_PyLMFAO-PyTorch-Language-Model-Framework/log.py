#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>
# Licensed under the Apache License v2.0 - http://www.apache.org/licenses/

"""
Log streamers
"""

import logging
import os
import configs

_log_file = '{}/{}.log'.format(configs.log_dir, configs.timestamp)


def init_logger():
    """
    Logger initializer
    Returns:
        logger
    """
    if not os.path.exists(configs.log_dir):
        try:
            os.mkdir(configs.log_dir)
        except IOError:
            print("Can not create log file directory.")
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    fh = logging.FileHandler(_log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def p_logger(s, print_=True, log_=True):
    """
    Print function redirection
    Args:
        s: stream text
        print_: flag, will not print on screen if set to False
        log_: flag, will not write into log file if set to False

    Returns:
        None
    """
    if print_:
        print(s)
    if log_:
        with open(_log_file, 'a+') as f_log:
            f_log.write(s + '\n')
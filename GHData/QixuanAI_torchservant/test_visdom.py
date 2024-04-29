# -*- coding: utf-8 -*-
# @Time    : 2019/09/16 01:41
# @Author  : LQX
# @Email   : qxsoftware@163.com
# @File    : test_visdom.py
# @Software: PyCharm

from torch_engine.core import BasicConfig
from torch_engine.utils.visdomtools import *


def main(args):
    config = BasicConfig()
    vis = VisdomVisualizer(config)
    vis.plot(1, 1, 'a', 'w')
    vis.plot(2, 2, 'a', 'w')


if __name__ == "__main__":
    import argparse

    parse = argparse.ArgumentParser()
    args = parse.parse_args()
    main(args)

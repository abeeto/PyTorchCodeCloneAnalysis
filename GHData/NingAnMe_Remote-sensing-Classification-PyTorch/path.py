# -*- coding: utf-8 -*
import sys

import os

DATA_ROOT_PATH = os.path.join(sys.path[0], 'data')
# 训练数据的路径
DATA_PATH = os.path.join(sys.path[0], 'data', 'input')
# 模型保存的路径
MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model', 'model.pkl')
# 训练log的输出路径
LOG_PATH = os.path.join(sys.path[0], 'data', 'output', 'logs')

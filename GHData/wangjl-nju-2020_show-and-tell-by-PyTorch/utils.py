import os
import random
from typing import List

import numpy as np
import torch


def set_seed(seed):
    """
    固定随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_file_from_paths(file_name, dirs: List):
    for each_dir in dirs:
        path = os.path.join(each_dir, file_name)
        if os.path.exists(path):
            return path
    print('{} not in {}.'.format(file_name, dirs))
    raise FileNotFoundError

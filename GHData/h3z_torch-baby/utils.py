import random
import tempfile

import torch

from config import RANDOM_STATE


def fix_random():
    random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)


def mktemp(f):
    return f"{tempfile.mkdtemp()}/{f}"

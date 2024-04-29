import datetime
import torch
from os.path import isdir
from shutil import rmtree
from os import makedirs


def force_create_dir(path):
    if isdir(path):
        rmtree(path)
    makedirs(path)


def get_readable_timestamp():
    stamp = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    stamp = stamp.replace(" ", "_")
    stamp = stamp.replace(":", "_")
    stamp = stamp.replace("-", "_")
    return stamp


def get_total_elements_cnt(x):
    tensor_shape = x.size()
    count = 1
    for dim_size in tensor_shape:
        count *= dim_size
    return count


class GrayToRgb(object):

    def __call__(self, sample):
        c, h, w = sample.size()
        out = torch.zeros(size=(3, h, w))
        out[0, ...] = sample[0].clone()
        out[1, ...] = sample[0].clone()
        out[2, ...] = sample[0].clone()
        return out

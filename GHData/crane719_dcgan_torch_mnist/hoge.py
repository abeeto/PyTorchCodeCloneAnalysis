import glob
import os
from collections import OrderedDict
import collections
import matplotlib.pyplot as plt
import numpy as np
import shutil

def try_gpu(obj):
    import torch
    if torch.cuda.is_available():
        return obj.cuda()
    return obj

def methods(obj):
    for method in dir(obj):
        print(method)

def get_filedirs(required_dir):
    return glob.glob(required_dir)

def make_dir(required_dirs):
    dirs = glob.glob("*")
    for required_dir in required_dirs:
        if not required_dir in dirs:
            print("generate file in current dir...")
            print("+ "+required_dir)
            os.mkdir(required_dir)

def recreate_dir(directory):
    for dir in directory:
        shutil.rmtree(dir)
    make_dir(directory)

def is_dir_existed(directory):
    dirs = glob.glob("*")
    if directory in dirs:
        return True
    else:
        return False

def flatten(nested_list):
    result = []
    for element in nested_list:
        if isinstance(element, collections.Iterable) and not isinstance(element, str):
            result.extend(flatten(element))
        else:
            result.append(element)
    return result

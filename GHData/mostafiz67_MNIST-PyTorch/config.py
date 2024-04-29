"""
Author: Md Mostafizur Rahman
File: Configaration file
"""

import os

nb_train_samples = 60000
nb_test_samples = 10000
nb_classes = 10

img_size = 28
img_channel = 1
img_shape = (img_size,  img_size,  img_channel)

lr = 0.01
batch_size = 2
nb_epochs = 1

def root_path():
    return os.path.dirname(__file__)

def checkpoints_path():
    return os.path.join(root_path(), "checkpoints")

def dataset_path():
    return os.path.join(root_path(), "dataset")

def output_path():
    return os.path.join(root_path(), "output")

def src_path():
    return os.path.join(root_path(), "src")

def submission_path():
    return os.path.join(root_path(), "submission")





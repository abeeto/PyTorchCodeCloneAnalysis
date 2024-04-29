from __future__ import print_function
# DataFrame
import pandas as pd
import gc
# import example_helper
import json
import csv
import dask.dataframe as dd
# Matplot
import matplotlib.pyplot as plt
# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import time
import pickle
import itertools
from itertools import islice
from torchmoji.model_def import torchmoji_transfer
from torchmoji.global_variables import PRETRAINED_PATH
from torchmoji.finetuning import (
     load_benchmark,
     finetune)
    


DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
dataset_path = 'data/training.10000.processed.noemoticon.csv'
print("Open file:", dataset_path)
data = pd.read_csv(dataset_path, encoding = DATASET_ENCODING, names = DATASET_COLUMNS)
print("Dataset size:", len(data))

decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}

def decode_sentiment(label):
    return decode_map[int(label)]
# %%time
data.target = data.target.apply(lambda x: decode_sentiment(x))

target_cnt = Counter(data.target)

plt.figure(figsize=(16,8))
plt.bar(target_cnt.keys(), target_cnt.values())
plt.title("Dataset labels distribuition")
plt.show()
# nb_classes = 2

# with open('../model/vocabulary.json', 'r') as f:
#     vocab = json.load(f)

# # Load dataset. Extend the existing vocabulary with up to 10000 tokens from
# # the training dataset.
# data = load_benchmark(DATASET_PATH, vocab, extend_with=10000)

# # Set up model and finetune. Note that we have to extend the embedding layer
# # with the number of tokens added to the vocabulary.
# model = torchmoji_transfer(nb_classes, PRETRAINED_PATH, extend_embedding=data['added'])
# print(model)
# model, acc = finetune(model, data['texts'], data['labels'], nb_classes,
#                       data['batch_size'], method='chain-thaw')
# print('Acc: {}'.format(acc))
